'''
Run this file with:
python bug.py -m configs/sweep.yaml
'''

from typing import TypeVar, Tuple, Any, Optional
import hydra
from omegaconf import DictConfig

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.static import static_data, static_functions
from mechagogue.player_list import birthday_player_list, player_family_tree

from dirt.constants import (
    DEFAULT_FLOAT_DTYPE, DEFAULT_BUG_COLOR, PHOTOSYNTHESIS_COLOR)
import dirt.gridworld2d.dynamics as dynamics
import dirt.gridworld2d.spawn as spawn
import dirt.gridworld2d.grid as grid
from dirt.gridworld2d.observations import noisy_sensor
from dirt.distribution.stochastic_rounding import stochastic_rounding

'''
Rough notes:
Expended energy when moving:
    mass (biomass + water) *
    manhattan_distance_travelled (based on movement_primitive trait) *
    land/water_movement_efficiency (trait) *
    movement_energy (param)

Expended energy when reproducing:
    baby_mass = (baby_biomass + baby_water) (trait)
    energy_per_birth_mass (param)

Biomass requirements:
    min_biomass_constant (params) +
    biomass_required_for_brain = (
        brain_size (trait) + biomass_per_brain_size (param)) +
    biomass_required_for_motion_primitives +
    biomass_required_for_attack_primitives (maybe area * abs(damage) + distance)
        (ooh! do we allow damage to be negative so agents can heal others?)
    
    Vincent: Why not! Think won't hurt the logic as a whole but will make the world more interesting
    
If biomass requirement not met (agent is born too small):
    lose health? can't do things?  Is there a way we can make it so that the
    babies can start small and grow big without dying right away?  They would
    have to eat biomass to grow to adult size.  How do we use traits to allow
    for development?  How do we charge the parent the correct ammount of
    biomass for their baby, when mutation could cause the requirements to
    change?  Also, if we allow for some kind of development, when does that
    happen?  Ok, one option: as a parameter, we have N "levels" per agent.
    Each level has it's own set of movement primitives, etc. so that each level
    will have its own biomass cost.  Then a player can take an action to switch
    to another level, but it only works if it has enough resources.  Ok, so
    that's an easy way to do development.  Still the question: what do we do if
    a player doesn't have enough biomass?  OR we could make sure that it always
    does have enough biomass by making level 0 the same for all agents, which
    is set as a parameter, and then if agents want to level up, they have to
    get the resources to do it, and levels 1-N are traits that can all be
    mutated, etc.  That's... fine?  Oh, or parent's have trait, which is Baby's
    first level that MUST get copied verbatim (no mutation) to baby's first
    level?  Eeh?  Or we keep around the parent's traits to handle this instead
    of copying?
    
    Ok, after thinking for a moment, I think we need to just make agents take
    damage (rapidly) if they don't have the right biomass.  Because this lets
    us handle agents that are allowed to expell biomass too (which we would
    have to clip if we()ere doing this in a constraint-based setup.
    
    I guess the distinction is between constraint-based and penalty-based
    rules for this.  The argument for penalty-based is that it is very easy.
    Agent's can have a "baby size" trait, and then if the actual baby's size
    doesn't match, it dies pretty quickly.  The downsize is that it may be
    possible to "hack" the system and allow for agents that have less biomass
    than should be possible, and use the associated skills and features, and
    make up for it by healing themselves and getting the biomass later.  I
    guess you can make this short-lived by killing them at a faster rate than
    they can heal?  Using constraints is kind of nice because you don't get
    hacking, but it sucks because you have to jump through extra hoops, and
    have to know something about the new child's traits when deciding if
    an agent is able to reproduce.  So yeah let's do penalty-based.

Do we allow agents to expell biomass/energy/water to either share with others
or reduce mass?
'''

# constants
NO_ACTION_TYPE = 0
MOVE_ACTION_TYPE = 1
ATTACK_ACTION_TYPE = 2
EAT_ACTION_TYPE = 3
EXPELL_ACTION_TYPE = 4
SMELL_ACTION_TYPE = 5
AUDIO_ACTION_TYPE = 6
LEVEL_ACTION_TYPE = 7
REPRODUCE_ACTION_TYPE = 8
NUM_ACTION_TYPES = 9

action_type_names = {
    0 : 'no action',
    1 : 'move',
    2 : 'attack',
    3 : 'eat',
    4 : 'expell',
    5 : 'emit smell',
    6 : 'emit audio',
    7 : 'change level',
    8 : 'reproduce',
}

@static_data
class BugParams:
    world_size : Tuple[int,int] = (1024,1024)
    initial_players : int = 1024
    max_players : int = 16384
    
    include_water : bool = True
    include_energy : bool = True
    include_biomass : bool = True
    
    levels : int = 0
    
    # costs
    # - resting
    resting_water_cost : float = 0.00005
    resting_water_cost_per_mass : float = 0.0001
    resting_energy_cost : float = 0.0005
    resting_energy_cost_per_mass : float = 0.001
    # - brain
    brain_size_water_cost : float = 0.0001
    brain_size_energy_cost : float = 0.001
    # - movement
    move_water_cost : float = 0.0001
    move_energy_cost : float = 0.001
    climb_water_cost : float = 0.0001
    climb_energy_cost : float = 0.001
    # - attack
    attack_energy_cost : float = 0.01
    # - reproduction
    birth_energy_cost : float = 0.001
    birth_water_cost : float = 0.0001
    birth_damage : float = 1.
    
    # biomass requirements
    # Vincent first pass for biomass requirements of various traits, random put some parameters here
    min_biomass_constant : float = 0.1
    biomass_per_brain_size : float = 0.05
    biomass_per_movement_primitive : float = 0.02
    biomass_per_attack_primitive : float = 0.03

    biomass_per_photosynthesis : float = 0.01
    biomass_per_max_climb : float = 0.01

    biomass_per_max_water : float = 0.01
    biomass_per_water_gulp : float = 0.01
    biomass_per_max_energy : float = 0.01
    biomass_per_energy_gulp : float = 0.01
    biomass_per_max_biomass: float = 0.01
    biomass_per_biomass_gulp : float = 0.01

    biomass_per_max_hp : float = 0.01

    # damage paramters
    lack_biomass_damage: float = 10.0

    # mass
    biomass_mass : float = 1.
    water_mass : float = 1.
    
    photosynthesis_energy_gain : float = 0.005
    
    # actions
    movement_primitives : int = 3
    attack_primitives : int = 1
    odor_primitives : int = 1
    audio_primitives : int = 1
    
    # health
    #max_hp_per_mass : float = 10.
    hp_per_armor : float = 10.
    hp_healed_per_energy : float = 10.
    water_underpayment_damage : float = 10000.
    energy_underpayment_damage : float = 1000.
    
    # initialization
    initial_hp : float = 10.
    initial_water : float = 0.1
    initial_energy : float = 1.
    initial_biomass : float = 1.0
    #initial_color : Tuple[float, float, float] = (0.5, 0.25, 0.5)
    
    # trait mutation parameters
    # - mutation noise rates
    # -- global
    mutate_trait_probability : float = 0.1 
    mutate_default_noise : float = 0.05
    mutate_default_lognoise : float = 0.05
    # -- specific
    #mutate_max_age_observation_noise : float = 1.
    #mutate_view_distance_noise : float = 0.1
    #mutate_view_back_distance_noise : float = 0.1
    #mutate_view_width_noise : float = 0.2
    #mutate_max_altitude_observation_noise : float = 0.1
    #mutate_max_water_observation_noise : float = 0.1
    #mutate_max_energy_observation_noise : float = 0.1
    #mutate_max_biomass_observation_noise : float = 0.1
    #
    #mutate_min_temperature_observation_noise : float = 0.1
    #mutate_max_temperature_observation_noise : float = 0.1
    #mutate_max_climb_noise : float = 0.1
    #mutate_max_water_noise : float = 0.1
    #mutate_max_water_gulp_noise : float = 0.1
    #mutate_max_energy_noise : float = 0.1
    #mutate_max_energy_gulp_noise : float = 0.1
    #mutate_max_biomass_noise : float = 0.1
    #mutate_max_biomass_gulp_noise : float = 0.1
    #
    #mutate_armor_noise : float = 0.1
    #mutate_senescence_damage_noise : float = 0.1
    #mutate_max_hp_noise : float = 0.1
    #mutate_healing_rate_noise : float = 0.1
    #
    #mutate_child_water_noise : float = 0.1
    #mutate_child_hp_noise : float = 0.1
    #mutate_child_energy_noise : float = 0.1
    #mutate_child_biomass_noise : float = 0.1
    #
    #mutate_movement_noise : float = 0.1
    
    # - min and max values
    max_view_distance : int = 5
    max_view_back_distance : int = 5
    max_view_width : int = 11
    max_max_altitude_observation : float = 5.
    max_max_water_observation : float = 5.
    max_max_energy_observation : float = 5.
    max_max_biomass_observation : float = 5.
    min_min_temperature_observation : float = -5.
    max_min_temperature_observation : float = 5.
    min_max_temperature_observation : float = -5.
    max_max_temperature_observation : float = 5.
    max_max_climb : float = 5.
    max_max_water : float = 10.
    max_max_water_gulp : float = 10.
    max_max_energy : float = 10.
    max_max_energy_gulp : float = 10.
    max_max_biomass : float = 10.
    max_max_biomass_gulp : float = 10.
    
    max_armor : float = 10.
    max_senescence_damage : float = 100.
    max_max_hp : float = 100.
    max_healing_rate : float = 10.
    
    max_child_water : float = 10.
    max_child_hp : float = 10.
    max_child_energy : float = 10.
    max_child_biomass : float = 10.
    
    max_movement : float = 4.
    
    def validate(params):
        assert params.initial_players <= params.max_players, (
            f'params.initial_players ({params.initial_players}) must be less '
            f'than or equal to params.max_players ({params.max_players})'
        )
        
        return params

@static_data
class BugTraits:
    # brain
    brain_size : float | jnp.ndarray
    
    # color
    color : Tuple[float, float, float] | jnp.ndarray
    
    # resources
    photosynthesis : float | jnp.ndarray
    
    # sensing
    # - age
    max_age_observation : float | jnp.ndarray
    age_sensor_noise : float | jnp.ndarray
    # - visual
    view_distance : float | jnp.ndarray
    view_back_distance : int | jnp.ndarray
    view_width : int | jnp.ndarray
    max_altitude_observation : float | jnp.ndarray
    visual_sensor_noise : float | jnp.ndarray
    # - audio
    audio_sensor_noise : float | jnp.ndarray
    # - smell
    smell_sensor_noise : float | jnp.ndarray
    # - external resources
    max_water_observation : float | jnp.ndarray
    max_energy_observation : float | jnp.ndarray
    max_biomass_observation : float | jnp.ndarray
    external_resource_sensor_noise : float | jnp.ndarray
    # - wind
    wind_sensor_noise : float | jnp.ndarray
    # - temperature
    min_temperature_observation : float | jnp.ndarray
    max_temperature_observation : float | jnp.ndarray
    temperature_sensor_noise : float | jnp.ndarray
    # - health and internal resources
    health_sensor_noise : float | jnp.ndarray
    internal_resource_sensor_noise : float | jnp.ndarray
    
    # efficiency
    max_climb : float | jnp.ndarray
    #climb_efficiency : float | jnp.ndarray
    #move_effciency : float | jnp.ndarray
    
    # stomach
    max_water : float | jnp.ndarray
    water_gulp : float | jnp.ndarray
    max_energy : float | jnp.ndarray
    energy_gulp : float | jnp.ndarray
    max_biomass : float | jnp.ndarray
    biomass_gulp : float | jnp.ndarray
    
    # climate
    insulation : float | jnp.ndarray
    
    # armor
    armor : float | jnp.ndarray
    
    # age
    senescence_damage : float | jnp.ndarray
    
    # health
    max_hp : float | jnp.ndarray
    healing_rate : float | jnp.ndarray
    
    # reproduction
    child_hp : float | jnp.ndarray
    child_water : float | jnp.ndarray
    child_energy : float | jnp.ndarray
    child_biomass : float | jnp.ndarray
    child_color : Tuple[float, float, float] | jnp.ndarray
    
    # actions
    movement_primitives : jnp.ndarray
    attack_primitives : jnp.ndarray
    
    @staticmethod
    def default(shape):
        if isinstance(shape, int):
            shape = (shape,)
        def float_vector(v):
            return jnp.full(shape, v, dtype=DEFAULT_FLOAT_DTYPE)
        
        def int_vector(v):
            return jnp.full(shape, v, dtype=jnp.int32)
        
        return BugTraits(
            # brain
            brain_size = float_vector(0.),
            
            # color
            color = jnp.full(
                (*shape,3), DEFAULT_BUG_COLOR, dtype=DEFAULT_FLOAT_DTYPE),
            
            # resources
            photosynthesis = float_vector(0.),
            
            # sensing
            # - age
            max_age_observation = float_vector(1000.),
            age_sensor_noise = float_vector(0.),
            # - visual
            view_distance = float_vector(5),
            view_back_distance = float_vector(0),
            view_width = float_vector(5),
            max_altitude_observation = float_vector(1.),
            visual_sensor_noise = float_vector(0.),
            # - audio
            audio_sensor_noise = float_vector(0.),
            # - smell
            smell_sensor_noise = float_vector(0.),
            # - external resources
            max_water_observation = float_vector(1.),
            max_energy_observation = float_vector(1.),
            max_biomass_observation = float_vector(1.),
            external_resource_sensor_noise = float_vector(0.),
            # - wind
            wind_sensor_noise = float_vector(0.),
            # - temperature
            min_temperature_observation = float_vector(-3.),
            max_temperature_observation = float_vector(3.),
            temperature_sensor_noise = float_vector(0.),
            # internal resources
            health_sensor_noise = float_vector(0.),
            internal_resource_sensor_noise = float_vector(0.),
            
            # efficiency
            max_climb = float_vector(2.),
            #climb_efficiency = float_vector(1.),
            #move_effciency = float_vector(1.),
            
            # stomach
            max_water = float_vector(1.),
            water_gulp = float_vector(0.5),
            max_energy = float_vector(2.),
            energy_gulp = float_vector(0.5),
            max_biomass = float_vector(5.),
            biomass_gulp = float_vector(0.5),
            
            # climate
            insulation = float_vector(0.),
            
            # armor
            armor = float_vector(0.),
            
            # age
            senescence_damage = float_vector(0.00001),
            
            # health
            max_hp = float_vector(10.),
            healing_rate = float_vector(1.),
            
            # reproduction
            child_hp = float_vector(5.),
            child_energy = float_vector(1.0),
            child_biomass = float_vector(1.0),
            child_water = float_vector(0.5),
            child_color = jnp.full(
                (*shape,3), DEFAULT_BUG_COLOR, dtype=DEFAULT_FLOAT_DTYPE),
            
            # actions
            movement_primitives = jnp.full(
                (*shape, 3, 3), jnp.array([
                    [1, 0, 0],
                    [0, 0,-1],
                    [0, 0, 1],
                ], dtype=DEFAULT_FLOAT_DTYPE)
            ),
            attack_primitives = jnp.full(
                (*shape, 1, 5), jnp.array([
                    [1, 0, 0, 0, 1]
                ], dtype=DEFAULT_FLOAT_DTYPE),
            )
            #attack_primitives = jnp.array([
            #    [1, 0, 0, 0, 1] # x-offset, y-offset, width, height, damage
            #], dtype=DEFAULT_FLOAT_DTYPE),
        )
    
    def biomass_requirement(self, params: "BugParams"):
        """
        Calculate the biomass requirement for this bug's traits.
        """
        n_move = self.movement_primitives.shape[0] if hasattr(self.movement_primitives, "shape") else 0
        n_attack = self.attack_primitives.shape[0] if hasattr(self.attack_primitives, "shape") else 0
        req = (
            params.min_biomass_constant
            + self.brain_size * params.biomass_per_brain_size
            + n_move * params.biomass_per_movement_primitive
            + n_attack * params.biomass_per_attack_primitive
            + self.photosynthesis * params.biomass_per_photosynthesis
            + self.max_climb * params.biomass_per_max_climb
            + self.max_water * params.biomass_per_max_water
            + self.water_gulp * params.biomass_per_water_gulp
            + self.max_energy * params.biomass_per_max_energy
            + self.energy_gulp * params.biomass_per_energy_gulp
            + self.max_biomass * params.biomass_per_max_biomass
            + self.biomass_gulp * params.biomass_per_biomass_gulp
            + self.max_hp * params.biomass_per_max_hp
        )
        return req

@static_data
class BugObservation:
    
    # age
    age : jnp.ndarray
    newborn : jnp.ndarray
    
    # visual
    rgb : jnp.ndarray
    relative_altitude : jnp.ndarray

    # audio
    audio : jnp.ndarray
    
    # smell
    smell : jnp.ndarray
    
    # weather
    wind : jnp.ndarray
    temperature : jnp.ndarray
    
    # external resources
    external_water : jnp.ndarray
    external_energy : jnp.ndarray
    external_biomass : jnp.ndarray
             
    # health and internal resources
    health : jnp.ndarray
    internal_water : jnp.ndarray
    internal_energy : jnp.ndarray
    internal_biomass : jnp.ndarray

#@static_data
#class BugAction:
#    move : jnp.ndarray
#    bite : jnp.ndarray
#    eat : jnp.ndarray
#    reproduce : jnp.ndarray

@static_data
class BugState:
    # location
    x : jnp.ndarray
    r : jnp.ndarray
    object_grid : jnp.ndarray
    
    # age
    age : jnp.ndarray
    
    # health
    hp : jnp.ndarray
    
    # resources
    water : jnp.ndarray
    energy : jnp.ndarray
    biomass : jnp.ndarray
    
    # color
    #color : jnp.ndarray
    
    # level
    level : jnp.ndarray
    
    # tracking
    family_tree : Any

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def make_bugs(
    params : BugParams = BugParams(),
    float_dtype : Any = DEFAULT_FLOAT_DTYPE
):
    
    params = params.validate()
    
    player_list = birthday_player_list(params.max_players)
    family_tree = player_family_tree(player_list, 1)
    
    # construct the action mapping
    # - figure out how many eat and expell action primitives there are
    eat_primitives = (
        int(params.include_water) +
        int(params.include_energy) +
        int(params.include_biomass)
    )
    expell_primitives = eat_primitives
    
    # - compute the total number of actions
    action_primitives = {
        NO_ACTION_TYPE : 1,
        MOVE_ACTION_TYPE : params.movement_primitives,
        ATTACK_ACTION_TYPE : params.attack_primitives,
        EAT_ACTION_TYPE : eat_primitives,
        EXPELL_ACTION_TYPE : expell_primitives,
        SMELL_ACTION_TYPE : params.odor_primitives,
        AUDIO_ACTION_TYPE : params.audio_primitives,
        LEVEL_ACTION_TYPE : params.levels,
        REPRODUCE_ACTION_TYPE : 1,
    }
    num_action_primitives = sum(action_primitives.values())
    max_action_primitives = max(action_primitives.values())
    
    # - build the mapping that goes from integers to action type and primitive
    #   index, and vice-versa
    action_to_primitive_map = jnp.zeros(
        (num_action_primitives, 2), dtype=jnp.int32)
    primitive_to_action_map = jnp.zeros(
        (len(action_primitives), max_action_primitives), dtype=jnp.int32)
    action_index = 0
    for action_type, actions_per_type in action_primitives.items():
        for j in range(actions_per_type):
            action_to_primitive_map = (
                action_to_primitive_map.at[action_index].set((action_type, j)))
            primitive_to_action_map = (
                primitive_to_action_map.at[action_type, j].set(action_index))
            action_index += 1
    
    # resource primitives
    resource_primitive = 0
    if params.include_water:
        WATER_PRIMITIVE = resource_primitive
        resource_primitive += 1
    if params.include_energy:
        ENERGY_PRIMITIVE = resource_primitive
        resource_primitive += 1
    if params.include_biomass:
        BIOMASS_PRIMITIVE = resource_primitive
        resource_primitive += 1
    
    @static_functions
    class Bugs:
        num_actions = num_action_primitives
        
        def init(
            key : chex.PRNGKey,
        ) -> BugState :
            
            # initialize the family tree
            family_tree_state = family_tree.init(params.initial_players)
            active_players = family_tree.active(family_tree_state)
            
            # initialize map positions and rotations
            key, xr_key = jrng.split(key)
            x, r = spawn.unique_xr(
                xr_key,
                params.max_players,
                params.world_size,
                active=active_players,
            )
            object_grid = jnp.full(params.world_size, -1, dtype=jnp.int32)
            object_grid = object_grid.at[x[...,0], x[...,1]].set(
                jnp.arange(params.max_players))
            
            # initialize the player age
            age = jnp.zeros((params.max_players,), dtype=jnp.int32)
            
            # initialize hp, water, energy, and biomass
            hp = (
                active_players*params.initial_hp).astype(float_dtype)
            if params.include_water:
                water = (
                    active_players*params.initial_water).astype(float_dtype)
            else:
                water = None
            if params.include_energy:
                energy = (
                    active_players*params.initial_energy).astype(float_dtype)
            else:
                energy = None
            if params.include_biomass:
                biomass = (
                    active_players*params.initial_biomass).astype(float_dtype)
            else:
                biomass = None
            
            # initialize color
            n = active_players.shape[0]
            #color = jnp.full(
            #    (n, 3), jnp.array(params.initial_color, dtype=float_dtype))
            
            # initialize level
            level = jnp.zeros(params.max_players, dtype=jnp.int32)
            
            return BugState(
                x=x,
                r=r,
                object_grid=object_grid,
                age=age,
                
                hp=hp,
                
                energy=energy,
                biomass=biomass,
                water=water,
                
                #color=color,
                
                level=level,
                
                family_tree=family_tree_state,
            )
        
        def get_mass(water, biomass):
            if params.include_biomass or params.include_water:
                mass = jnp.zeros((), dtype=float_dtype)
                if params.include_biomass:
                    mass += biomass * params.biomass_mass
                if params.include_water:
                    mass += water * params.water_mass
            else:
                mass = jnp.ones((params.max_population,), dtype=float_dtype)
            
            return mass
        
        def move(
            key : chex.PRNGKey,
            state : BugState,
            action : int,
            traits : BugTraits,
            altitude : jnp.ndarray,
            altitude_downsample : int,
        ):
            
            # validate
            assert (
                params.movement_primitives ==
                traits.movement_primitives.shape[1]
            )
            
            # starting position and rotation
            x0 = state.x
            r0 = state.r
            g0 = state.object_grid
            
            # compute the movement offset (dxr) for all bugs
            action_type = action_to_primitive_map[action, 0]
            action_primitive = action_to_primitive_map[action, 1]
            move = (action_type == MOVE_ACTION_TYPE)
            all_players = jnp.arange(params.max_players)
            player_movement_primitives = (
                traits.movement_primitives[all_players, action_primitive])
            direction = jnp.where(move[..., None], player_movement_primitives,0)
            dxr = stochastic_rounding(key, direction)
            
            # move the bugs
            active_bugs = family_tree.active(state.family_tree)
            x1, r1, _, g1 = dynamics.movement_step(
                x0,
                r0,
                dxr,
                active=active_bugs,
                check_collisions=True,
                object_grid=g0,
            )
            
            # compute the energy and water costs
            dx = dynamics.distance_x(x1, x0)
            dr = dynamics.distance_r(r1, r0)
            y0 = grid.read_grid_locations(
                altitude, x0, altitude_downsample, downsample_scale=False)
            y1 = grid.read_grid_locations(
                altitude, x1, altitude_downsample, downsample_scale=False)
            dy = y1 - y0
            uphill = jnp.where(dy > 0., dy, 0.)
            can_move = uphill <= traits.max_climb
            mass = Bugs.get_mass(state.water, state.biomass)
            if params.include_energy:
                energy_cost = mass * (
                    params.move_energy_cost * (dx + dr) +
                    params.climb_energy_cost * uphill
                )
                can_move &= energy_cost <= state.energy
            if params.include_water:
                water_cost = mass * (
                    params.move_water_cost * (dx + dr) +
                    params.climb_water_cost * uphill
                )
                can_move &= water_cost <= state.water
            
            # undo the motion of any bugs that can not pay the necessary costs
            # or cannot climb well enough
            x2 = jnp.where(can_move[...,None], x1, x0)
            r2 = jnp.where(can_move, r1, r0)
            g2 = dynamics.move_objects(x1, x2, g1)
            
            # update state position/rotation
            state = state.replace(
                x=x2,
                r=r2,
                object_grid=g2,
            )
            
            # charge the energy and water for each bug appropriately
            if params.include_energy:
                energy_cost = energy_cost * can_move * active_bugs
                energy = state.energy - energy_cost
                state = state.replace(energy=energy)
            if params.include_water:
                water_cost = water_cost * can_move * active_bugs
                water = state.water - water_cost
                state = state.replace(water=water)
            else:
                water_cost = None
            
            return state, water_cost
        
        def eat(
            state : BugState,
            action : int,
            traits : BugTraits,
            water : Optional[jnp.ndarray] = None,
            energy : Optional[jnp.ndarray] = None,
            biomass : Optional[jnp.ndarray] = None,
        ):
            # get the right traits
            if params.levels:
                traits = tree_getitem(traits, state.level)
            
            # break the action integers into action types and primitives
            action_type = action_to_primitive_map[action, 0]
            action_primitive = action_to_primitive_map[action, 1]
            
            # compute which bugs are eating anything
            active = Bugs.active_players(state)
            eat = (action_type == EAT_ACTION_TYPE) & active
            expell = (action_type == EXPELL_ACTION_TYPE) & active
            
            # water
            if params.include_water:
                # - compute which bugs are drinking water
                drink_water = eat & (action_primitive == WATER_PRIMITIVE)
                expell_water = expell & (action_primitive == WATER_PRIMITIVE)
                # - compute how much water each bug will consume
                desired_water = drink_water * jnp.minimum(
                    traits.water_gulp, traits.max_water - state.water)
                consumed_water = jnp.minimum(desired_water, water)
                # - compute how much water each bug will expell
                expelled_water = expell_water * jnp.minimum(
                    traits.water_gulp, state.water)
                # - compute the lefover water and the bugs' new water
                leftover_water = water - consumed_water + expelled_water
                new_water = state.water + consumed_water - expelled_water
                # - update state
                state = state.replace(water=new_water)
            else:
                leftover_water = None
            
            # energy
            if params.include_energy:
                # - compute which bugs are eating energy
                eat_energy = eat & (action_primitive == ENERGY_PRIMITIVE)
                expell_energy = expell & (action_primitive == ENERGY_PRIMITIVE)
                # - compute how much energy each bug will consume
                desired_energy = eat_energy * jnp.minimum(
                    traits.energy_gulp, traits.max_energy - state.energy)
                consumed_energy = jnp.minimum(desired_energy, energy)
                # - compute how much energy each bug will expell
                expelled_energy = expell_energy * jnp.minimum(
                    traits.energy_gulp, state.energy)
                # - compute the leftover energy and the bugs' new energy
                leftover_energy = energy - consumed_energy + expelled_energy
                new_energy = state.energy + consumed_energy - expelled_energy
                # - update state
                state = state.replace(energy=new_energy)
            else:
                leftover_energy = None
            
            # biomass
            if params.include_biomass:
                # - compute which bugs are eating biomass
                eat_biomass = eat & (action_primitive == BIOMASS_PRIMITIVE)
                expell_biomass = expell&(action_primitive == BIOMASS_PRIMITIVE)
                # - compute how much biomass each bug will consume
                desired_biomass = eat_biomass * jnp.minimum(
                    traits.biomass_gulp, traits.max_biomass - state.biomass)
                consumed_biomass = jnp.minimum(desired_biomass, biomass)
                # - compute how much biomass each bug will expell
                expelled_biomass = expell_biomass * jnp.minimum(
                    traits.biomass_gulp, state.biomass)
                # - compute the leftover biomass and the bugs' new energy
                leftover_biomass = biomass - consumed_biomass + expelled_biomass
                new_biomass = state.biomass + consumed_biomass -expelled_biomass
                # - update state
                state = state.replace(biomass=new_biomass)
            else:
                leftover_biomass = None
            
            return state, leftover_water, leftover_energy, leftover_biomass
        
        def photosynthesis(
            state : BugState,
            traits : BugTraits,
            light : jnp.ndarray,
        ):
            energy_gain = (
                traits.photosynthesis *
                params.photosynthesis_energy_gain *
                light
            )
            state = state.replace(energy=state.energy + energy_gain)
            return state
        
        def heal(
            state : BugState,
            traits : BugTraits,
        ):
            healable_hp = traits.max_hp - state.hp
            healing_energy_cost = healable_hp / params.hp_healed_per_energy
            usable_energy_cost = jnp.minimum(healing_energy_cost, state.energy)
            hp_to_heal = usable_energy_cost * params.hp_healed_per_energy
            state = state.replace(
                hp = state.hp + hp_to_heal,
                energy = state.energy - usable_energy_cost,
            )
            return state
        
        def fight(
            state : BugState,
            action : int,
            traits : BugTraits,
        ):
            assert (
                params.attack_primitives ==
                traits.attack_primitives.shape[0]
            )
            action_type = action_to_primitive_map[action, 0]
            action_primitive = action_to_primitive_map[action, 1]
            attack = (action_type == ATTACK_ACTION_TYPE)
            
            active_bugs = family_tree.active(state.family_tree)
            
            attack_offsets = jnp.where(
                attack[..., None],
                traits.attack_primitives[action_primitive],
                jnp.zeros_like(traits.attack_primitives[0])
            )
            attack_pos = state.x + attack_offsets
            bug_pos = state.x
            
            def generate_attack_offsets(dx, dy, width, height):
                wx = width
                wy = height
                x_range = jnp.arange(-wx, wx + 1)
                y_range = jnp.arange(-wy, wy + 1)
                grid_x, grid_y = jnp.meshgrid(x_range, y_range, indexing="xy")
                offsets = jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
                return offsets + jnp.array([dx, dy])

            def single_attack(
                attacker_idx,
                attack_flag,
                prim_idx,
                attacker_pos
            ):
                # If not attacking, return no damage
                def not_attacking():
                    return jnp.zeros(bug_pos.shape[0]), 0.0

                def attacking():
                    prim = traits.attack_primitives[prim_idx]
                    dx, dy, w, h, damage = prim

                    offsets = generate_attack_offsets(dx, dy, w, h)
                    attack_tiles = attacker_pos[None, :] + offsets

                    # Check which bugs are on any attack tile
                    def is_hit(target_pos):
                        return jnp.any(jnp.all(
                            attack_tiles == target_pos[None, :], axis=-1))

                    hits = jax.vmap(is_hit)(bug_pos)
                    hits = hits.at[attacker_idx].set(0) # prevent self hit
                    per_target_damage = hits * damage
                    energy_cost = params.attack_energy_cost
                    return per_target_damage, energy_cost

                return jax.lax.cond(attack_flag, attacking, not_attacking)

            damages, energy_costs = jax.vmap(
                single_attack,
                in_axes=(0, attack, action_primitive, bug_pos)
            )(jnp.arange(bug_pos.shape[0]), attack, action_primitive, bug_pos)

            damage_received = jnp.sum(damages, axis=0)
            new_hp = state.hp - damage_received

            if params.include_energy:
                energy_cost = params.attack_energy_cost * attack * active_bugs
                new_energy = state.energy - energy_cost
            else:
                new_energy = state.energy

            state = state.replace(
                hp=new_hp,
                energy=new_energy
            )

            return state
        
        def metabolism(
            state : BugState,
            traits : BugTraits,
        ):
            damage = jnp.zeros((params.max_players,), dtype=float_dtype)
            
            # change color
            #state = state.replace(color=traits.color)
            
            # energy cost
            if params.include_energy:
                alive = Bugs.active_players(state)
                mass = Bugs.get_mass(state.water, state.biomass)
                energy_cost = (
                    params.resting_energy_cost +
                    params.resting_energy_cost_per_mass * mass +
                    params.brain_size_energy_cost * traits.brain_size
                ) * alive
                energy_paid = jnp.minimum(state.energy, energy_cost)
                energy = state.energy - energy_paid
                energy_underpayment = energy_cost - energy_paid
                damage += (
                    energy_underpayment * params.energy_underpayment_damage)
            else:
                energy = None
            
            # water cost
            if params.include_water:
                water_cost = (
                    params.resting_water_cost +
                    params.resting_water_cost_per_mass * mass +
                    params.brain_size_water_cost * traits.brain_size
                ) * alive
                water_paid = jnp.minimum(state.water, water_cost)
                water = state.water - water_paid
                water_underpayment = water_cost - water_paid
                damage += water_underpayment * params.water_underpayment_damage
            else:
                water = None
                water_paid = None
            
            # senescence (aging)
            damage += traits.senescence_damage * state.age
            
            # Vincent's pass to check biomass requirements
            biomass_req = traits.biomass_requirement(params)
            # check if the biomass requirement is met
            lack_biomass = state.biomass < biomass_req
            # if not, apply damage
            # turning this off momentarily until we tune the parameters
            #damage += lack_biomass * params.lack_biomass_damage
                       
            state = state.replace(
                hp=state.hp - damage,
                energy=energy,
                water=water,
            )
            
            return state, water_paid
        
        def birth_and_death(
            state : BugState,
            action : int,
            traits : BugTraits,
        ):
            # determine who is active and alive
            active = Bugs.active_players(state)
            alive = state.hp > 0
            
            # pre-register locations where expelled resources will be placed
            expelled_x = state.x
            
            # compute which bugs will reproduce
            # - compute which bugs want to reproduce
            # -- break the action integers into action types and primitives
            action_type = action_to_primitive_map[action, 0]
            action_primitive = action_to_primitive_map[action, 1]
            # -- filter by those taking the reproduce action
            wants_to_reproduce = action_type == REPRODUCE_ACTION_TYPE
            # - compute the parents' birth resource requirements
            child_mass = Bugs.get_mass(
                traits.child_water, traits.child_biomass)
            birth_water_cost = child_mass * params.birth_water_cost
            birth_energy_cost = child_mass * params.birth_energy_cost
            birth_damage = child_mass * params.birth_damage
            birth_required_water = birth_water_cost + traits.child_water
            birth_required_energy = birth_energy_cost + traits.child_energy
            birth_required_biomass = traits.child_biomass
            # - compute which bugs are able to reproduce
            #   (those that can satisfy the birth costs)
            able_to_reproduce = jnp.ones((params.max_players,), dtype=jnp.bool)
            if params.include_water:
                able_to_reproduce &= (state.water >= birth_required_water)
            if params.include_energy:
                able_to_reproduce &= (state.energy >= birth_required_energy)
            if params.include_biomass:
                able_to_reproduce &= (state.biomass >= birth_required_biomass)
            # - compute which bugs will actually reproduce
            will_reproduce = alive & wants_to_reproduce & able_to_reproduce
            # - compute new positions and rotations for the children and
            #   make sure those new positions and rotations are unoccupied
            will_reproduce, child_x, child_r = spawn.spawn_from_parents(
                will_reproduce,
                state.x,
                state.r,
                object_grid=state.object_grid,
            )
            
            # charge the parents for the required birth resources
            paid_hp = birth_damage * will_reproduce
            state = state.replace(hp = state.hp - paid_hp)
            if params.include_water:
                paid_water = birth_required_water * will_reproduce
                state = state.replace(water = state.water - paid_water)
            if params.include_energy:
                paid_energy = birth_required_energy * will_reproduce
                state = state.replace(energy = state.energy - paid_energy)
            if params.include_biomass:
                paid_biomass = birth_required_biomass * will_reproduce
                state = state.replace(biomass = state.biomass - paid_biomass)
            
            # update the family tree
            # - get the locations of the new parents
            parent_locations, = jnp.nonzero(
                will_reproduce,
                size=params.max_players,
                fill_value=params.max_players,
            )
            # - compute recent deaths
            #   (the parent dying from birth damage does not prevent birth)
            alive = state.hp > 0
            recent_deaths = active & ~alive
            
            # - increment the age of alive bugs and zero the age of dead bugs
            age = (state.age+1) * alive
            # - step the family tree
            family_tree_state, child_locations, _ = family_tree.step(
                state.family_tree,
                recent_deaths,
                parent_locations[..., None],
            )
            active = family_tree.active(family_tree_state)
            # - update state
            state = state.replace(
                age = age,
                family_tree = family_tree_state,
            )
            
            # update the bug physical positions and rotations
            # - move the dead bugs off the map and set 0 rotation
            x = jnp.where(
                recent_deaths[:,None],
                jnp.array(params.world_size, dtype=jnp.int32),
                state.x,
            )
            r = state.r * active
            # - insert the child positions and rotations
            # -- get the child_x and child_r that were actually produced
            child_x = child_x[parent_locations]
            child_r = child_r[parent_locations]
            # -- insert these child_x and child_r values into the position
            #    and rotation vectors
            x = x.at[child_locations].set(child_x)
            r = r.at[child_locations].set(child_r)
            # - update the object grid
            object_grid = state.object_grid.at[x[...,0], x[...,1]].set(
                jnp.where(active, jnp.arange(params.max_players), -1))
            # - update the state
            state = state.replace(x=x, r=r, object_grid=object_grid)
            
            # everybody gets older
            # moved this earlier so that new bugs still have age=0
            #state = state.replace(age=(state.age + 1) * active)
            
            # compute the child hp and resources
            child_hp = traits.child_hp[parent_locations]
            state = state.replace(
                hp = state.hp.at[child_locations].set(child_hp))
            
            # update the resources
            # get the resources of the dead bugs, and the resources expended
            # in birth to return to the landscape
            if params.include_water:
                expelled_moisture = paid_water
                dead_water = state.water * recent_deaths
                child_water = traits.child_water[parent_locations]
                water = state.water * ~recent_deaths
                water = water.at[child_locations].set(child_water)
                #state = state.replace(water = state.water * ~recent_deaths)
                state = state.replace(water=water)
            else:
                expelled_moisture = None
                dead_water = None
            if params.include_energy:
                dead_energy = state.energy * recent_deaths
                child_energy = traits.child_energy[parent_locations]
                energy = state.energy * ~recent_deaths
                energy = energy.at[child_locations].set(child_energy)
                #state = state.replace(energy = state.energy * ~recent_deaths)
                state = state.replace(energy=energy)
            else:
                dead_energy = None
            if params.include_biomass:
                dead_biomass = state.biomass * recent_deaths
                child_biomass = traits.child_biomass[parent_locations]
                biomass = state.biomass * ~recent_deaths
                biomass = biomass.at[child_locations].set(child_biomass)
                #state = state.replace(biomass = state.biomass * ~recent_deaths)
                state = state.replace(biomass=biomass)
            
            ## - update the state
            #    water = state.water.at[child_locations].set(child_water),
            #    energy = state.energy.at[child_locations].set(child_energy),
            #    biomass = state.biomass.at[child_locations].set(child_biomass),
            #)
            
            # zero negative hp
            state = state.replace(hp = jnp.clip(state.hp, min=0.))
            
            return (
                state,
                expelled_x,
                expelled_moisture,
                dead_water,
                dead_energy,
                dead_biomass,
            )
        
        def observe(
            key : chex.PRNGKey,
            state : BugState,
            traits : BugTraits,
            rgb : jnp.ndarray,
            relative_altitude : jnp.ndarray,
            audio : jnp.ndarray,
            smell : jnp.ndarray,
            wind : jnp.ndarray,
            temperature : jnp.ndarray,
            external_water : jnp.ndarray,
            external_energy : jnp.ndarray,
            external_biomass : jnp.ndarray,
        ):
            
            # age
            key, age_key = jrng.split(key)
            sensor_age = jnp.clip(
                state.age/traits.max_age_observation, min=0., max=1.)
            sensor_age = noisy_sensor(
                age_key, sensor_age, traits.age_sensor_noise)
            newborn = state.age == 0
            
            # for all observation variables below, the values for newborns
            # will be zeroed out because the traits for them have not been
            # provided yet
            
            # vision
            # - rgb
            key, rgb_key = jrng.split(key)
            sensor_rgb = noisy_sensor(rgb_key, rgb, traits.visual_sensor_noise)
            sensor_rgb *= ~newborn[:,None,None,None]
            
            # - altitude
            sensor_relative_altitude = jnp.clip(
                relative_altitude/traits.max_altitude_observation[:,None,None],
                min=-1.,
                max=1.,
            )
            key, altitude_key = jrng.split(key)
            sensor_relative_altitude = noisy_sensor(
                altitude_key,
                relative_altitude,
                traits.visual_sensor_noise,
                minval=-1.,
                maxval=1.,
            )
            sensor_relative_altitude *= ~newborn[:,None,None]
            
            # audio
            key, audio_key = jrng.split(key)
            sensor_audio = noisy_sensor(
                audio_key, audio, traits.audio_sensor_noise)
            sensor_audio *= ~newborn[:,None]
            
            # smell
            key, smell_key = jrng.split(key)
            sensor_smell = noisy_sensor(
                smell_key, smell, traits.smell_sensor_noise)
            sensor_smell *= ~newborn[:,None]
            
            # weather
            key, wind_key, temperature_key = jrng.split(key, 3)
            # - wind
            sensor_wind = noisy_sensor(
                wind_key, wind, traits.wind_sensor_noise) * ~newborn[:,None]
            # - temperature
            temperature_range = (
                traits.max_temperature_observation -
                traits.min_temperature_observation
            )
            sensor_temperature = jnp.clip(
                (temperature - traits.min_temperature_observation) /
                temperature_range,
                min=0.,
                max=1.,
            )
            sensor_temperature = noisy_sensor(
                temperature_key,
                sensor_temperature,
                traits.temperature_sensor_noise,
            ) * ~newborn
            
            # external resources
            key, water_key, energy_key, biomass_key = jrng.split(key, 4)
            # - water
            if params.include_water:
                sensor_external_water = jnp.clip(
                    external_water/traits.max_water_observation, min=0., max=1.)
                sensor_external_water = noisy_sensor(
                    water_key,
                    sensor_external_water,
                    traits.external_resource_sensor_noise,
                ) * ~newborn
            else:
                sensor_external_water = None
            # - energy
            if params.include_energy:
                sensor_external_energy = jnp.clip(
                    external_energy/traits.max_energy_observation,
                    min=0.,
                    max=1.,
                )
                sensor_external_energy = noisy_sensor(
                    energy_key,
                    sensor_external_energy,
                    traits.external_resource_sensor_noise,
                ) * ~newborn
            else:
                sensor_external_energy = None
            # - biomass
            if params.include_biomass:
                sensor_external_biomass = jnp.clip(
                    external_biomass/traits.max_biomass_observation,
                    min=0.,
                    max=1.,
                )
                sensor_external_biomass = noisy_sensor(
                    biomass_key,
                    sensor_external_biomass,
                    traits.external_resource_sensor_noise,
                ) * ~newborn
            else:
                sensor_external_biomass = None
            
            # health and internal resources
            key, health_key, water_key, energy_key, biomass_key = jrng.split(
                key, 5)
            # - health
            sensor_health = state.hp / traits.max_hp
            sensor_health = noisy_sensor(
                health_key,
                sensor_health,
                traits.health_sensor_noise,
            )
            sensor_health *= ~newborn
            # - water
            if params.include_water:
                sensor_internal_water = state.water/traits.max_water
                sensor_internal_water = noisy_sensor(
                    water_key,
                    sensor_internal_water,
                    traits.internal_resource_sensor_noise,
                ) * ~newborn
            else:
                sensor_internal_water = None
            # - energy
            if params.include_energy:
                sensor_internal_energy = state.energy/traits.max_energy
                sensor_internal_energy = noisy_sensor(
                    energy_key,
                    sensor_internal_energy,
                    traits.internal_resource_sensor_noise,
                ) * ~newborn
            else:
                sensor_internal_energy = None
            # - biomass
            if params.include_biomass:
                sensor_internal_biomass = state.biomass/traits.max_biomass
                sensor_internal_biomass = noisy_sensor(
                    biomass_key,
                    sensor_internal_biomass,
                    traits.internal_resource_sensor_noise,
                ) * ~newborn
            else:
                sensor_internal_biomass = None
            
            return BugObservation(
                age=sensor_age,
                newborn=newborn,
                rgb=sensor_rgb,
                relative_altitude=sensor_relative_altitude,
                audio=sensor_audio,
                smell=sensor_smell,
                external_water=sensor_external_water,
                external_energy=sensor_external_energy,
                external_biomass=sensor_external_biomass,
                wind=sensor_wind,
                temperature=sensor_temperature,
                health=sensor_health,
                internal_water=sensor_internal_water,
                internal_energy=sensor_internal_energy,
                internal_biomass=sensor_internal_biomass,
            )
        
        def get_action_type_and_primitive(action):
            return action_to_primitive_map[action]
    
        def mutate_traits(key, traits):
            def normal_mutate_trait(key, traits, trait_name):
                do_key, noise_key = jrng.split(key)
                do_mutate = jrng.bernoulli(
                    do_key, p=params.mutate_trait_probability, shape=())
                trait = getattr(traits, trait_name)
                max_trait = getattr(params, f'max_{trait_name}', 1.)
                min_trait = getattr(params, f'min_{trait_name}', 0.)
                noise_std = getattr(
                    params,
                    f'mutate_{trait_name}_noise',
                    params.mutate_default_noise * (max_trait - min_trait),
                )
                noise = jrng.normal(
                    noise_key, shape=trait.shape, dtype=float_dtype
                ) * noise_std * do_mutate
                trait = trait + noise
                
                traits = traits.replace(**{trait_name : trait+noise})
                return traits
            
            all_trait_names = set(
                key for key, value in traits.__dict__.items()
                if not callable(value)
            )
            nonstandard_trait_names = set((
                'brain_size',
                'max_age_observation',
                'senescence_damage',
                'movement_primitives',
                'attack_primitives',
            ))
            standard_trait_names = all_trait_names - nonstandard_trait_names
            
            for trait_name in standard_trait_names:
                key, trait_key = jrng.split(key)
                traits = normal_mutate_trait(trait_key, traits, trait_name)
            
            def lognormal_mutate_trait(key, traits, trait_name):
                do_key, noise_key = jrng.split(key)
                do_mutate = jrng.bernoulli(
                    do_key, p=params.mutate_trait_probability, shape=())
                trait = getattr(traits, trait_name)
                log_noise_std = getattr(
                    params,
                    f'mutate_{trait_name}_noise',
                    params.mutate_default_lognoise,
                )
                log_noise = jrng.normal(
                    noise_key, shape=(), dtype=float_dtype) * log_noise_std
                new_trait = jnp.exp(
                    jnp.log(trait) + log_noise * do_mutate)
                traits = traits.replace(**{trait_name:new_trait})
                return traits
            
            key, max_age_observation_key = jrng.split(key)
            traits = lognormal_mutate_trait(
                max_age_observation_key, traits, 'max_age_observation')
            key, senescence_damage_key = jrng.split(key)
            traits = lognormal_mutate_trait(
                senescence_damage_key, traits, 'senescence_damage')
            
            # move primitives
            key, do_key, primitive_key, dx_noise_key, dr_noise_key = (
                jrng.split(key, 5))
            do_mutate = jrng.bernoulli(
                do_key, p=params.mutate_trait_probability, shape=())
            mutate_primitive = jrng.choice(
                primitive_key, params.movement_primitives, shape=())
            dx_noise_std = getattr(
                params,
                f'mutate_movement_noise',
                params.mutate_default_noise * 2 * params.max_movement,
            )
            dx_noise = (
                jrng.normal(dx_noise_key, shape=(2,), dtype=float_dtype) *
                dx_noise_std *
                do_mutate
            )
            dr_noise_std = getattr(
                params,
                f'mutate_rotation_noise',
                params.mutate_default_noise * 4,
            )
            dr_noise = (
                jrng.normal(dr_noise_key, shape=(1,), dtype=float_dtype) *
                dr_noise_std *
                do_mutate
            )
            #jax.debug.print('dx noise std {dx}', dx=dx_noise_std)
            #jax.debug.print('dr noise std {dr}', dr=dr_noise_std)
            dx = traits.movement_primitives[:,:2]
            dx = dx.at[mutate_primitive].add(dx_noise)
            dx = jnp.clip(dx, min=-params.max_movement, max=params.max_movement)
            dr = traits.movement_primitives[:,[3]] + dr_noise
            dr = dr.at[mutate_primitive].add(dr_noise)
            dr = dr % 4
            movement_primitives = jnp.concatenate((dx, dr), axis=-1)
            
            traits = traits.replace(movement_primitives=movement_primitives)
            
            # TODO: Attack primitives
            
            return traits
        
        def metabolism_old(
            state : BugState,
            action : int,
            #next_state : BugState,
            traits : BugTraits,
            #water : jnp.ndarray,
            #light : jnp.ndarray,
        ):
        
            mass = Bugs.get_mass(state.water, state.biomass)
            alive = active_players(next_state)
            
            # compute energy costs
            energy_cost = jnp.zeros(params.max_players, dtype=float_dtype)
            # - resting energy cost
            energy_cost += params.resting_energy_cost * mass
            # - brain energy cost
            energy_cost += params.brain_size_energy_cost * traits.brain_size
            
            # - compute the updated energy
            #   (do no more energy changes after this point)
            next_energy = state.energy - energy_cost
            next_energy_clipped = jnp.where(next_energy < 0., 0., next_energy)
            
            # - compute the updated water
            #   (no no more water changes after this point)
            #next_water = next_stomach.water + water_offset
            #next_water_clipped = jnp.where(next_water < 0., 0., next_water)
            
            # - damage due to using more energy than is available
            #health_offset += jnp.where(
            #    next_energy < 0., next_energy, 0.)
            
            # - damage due to using more water than is available
            #health_offset += jnp.where(next_water < 0., next_water, 0.)
            
            # - damage due to having zero water
            #health_offset += jnp.where(
            #    next_water <= 0., -params.zero_water_damage, 0)
            
            # reproduce
            # - compute which agents are able to reproduce
            '''
            has_enough_energy_to_reproduce = (
                next_stomach.energy >=
                params.starting_energy + params.birth_energy
            )
            has_enough_biomass_to_reproduce = (
                next_stomach.biomass >= params.starting_biomass * 2)
            has_enough_water_to_reproduce = (
                next_stomach.water >=
                params.starting_water + params.birth_water
            )
                
            reproduce = (
                alive &
                action.reproduce &
                has_enough_energy_to_reproduce &
                has_enough_biomass_to_reproduce &
                has_enough_water_to_reproduce 
            )
            '''
            
            # - compute the reproduce positions and rotations
            reproduce, child_x, child_r = spawn.spawn_from_parents(
                reproduce,
                x1,
                r1,
                object_grid=next_state.object_grid,
            )
            n = reproduce.shape[0]
            parent_locations, = jnp.nonzero(reproduce, size=n, fill_value=n)
            parent_locations = parent_locations[..., None]
            child_x = child_x[parent_locations[...,0]]
            child_r = child_r[parent_locations[...,0]]
            
            # compute the birth costs 
            energy_offset -= params.birth_energy * reproduce
            water_offset -= params.birth_water * reproduce
            health_offset -= params.birth_damage * reproduce
            
            # compute the resources donated to the child
            energy_offset -= params.starting_energy * reproduce
            water_offset -= params.starting_water * reproduce
            biomass_offset -= params.starting_biomass * reproduce
            
            # - update the energy in the stomach and the health
            next_stomach = next_stomach.replace(
                energy=next_energy_clipped,
                water=next_water_clipped,
            )
            next_health = next_state.health + health_offset
            
            # senescence
            next_health = next_health * (1. - params.senescence)**next_state.age
            
            # color
            color = (
                traits.base_color * (1. - traits.photosynthesis[:,None]) + 
                PHOTOSYNTHESIS_COLOR * (traits.photosynthesis[:,None])
            )
            
            # deaths
            # - figure out who died
            deaths = next_health <= 0.
            # - update the next state variables appropriately
            next_health *= ~deaths
            
            xx, yy = x1[...,0], x1[...,1]
            next_object_grid = next_state.object_grid.at[xx, yy].set(
                jnp.where(deaths, -1, jnp.arange(deaths.shape[0])))
            
            '''
            expelled = Consumable(
                water=next_stomach.water * deaths,
                energy=next_stomach.energy * deaths,
                biomass=next_stomach.biomass * deaths,
            )
            expelled_locations = x1
            
            next_stomach = next_stomach.replace(
                water=next_stomach.water * ~deaths,
                energy=next_stomach.energy * ~deaths,
                biomass=next_stomach.biomass * ~deaths,
            )
            '''
            expelled_energy = SOMETHING * deaths
            expelled_biomass = SOMETHING * deaths
            expelled_water = SOMETHING * deaths
            
            x2 = jnp.where(
                deaths[:,None],
                jnp.array(params.world_size, dtype=jnp.int32),
                x1,
            )
            
            n = next_state.family_tree.parents.shape[0]
            next_family_tree_state, children, _ = family_tree.step(
                next_state.family_tree,
                deaths,
                parent_locations,
            )
            next_alive = family_tree.active(next_family_tree)
            next_age = (next_state.age + 1)*next_alive
            
            # update the children's age, water, energy, biomass, and health
            next_age.at[children].set(0)
            child_energy = next_stomach.energy.at[children].set(
                params.starting_energy)
            child_biomass = next_stomach.biomass.at[children].set(
                params.starting_biomass)
            child_water = next_stomach.water.at[children].set(
                params.starting_water)
            next_stomach = next_stomach.replace(
                energy=child_energy,
                biomass=child_biomass,
                water=child_water,
            )
            next_health = next_health.at[children].set(params.starting_health)
            x2 = x2.at[children].set(child_x)
            r1 = r1.at[children].set(child_r)
            
            child_xx, child_yy = child_x[...,0], child_x[...,1]
            next_object_grid = (
                next_object_grid.at[child_xx, child_yy].set(children))
            
            next_state = next_state.replace(
                x=x2,
                r=r1,
                stomach=next_stomach,
                health=next_health,
                color=color,
                family_tree=next_family_tree_state,
                age=next_age,
                object_grid = next_object_grid,
            )
            
            return next_state, expelled, expelled_locations
            
        def active_players(
            state : BugState,
        ):
            return family_tree.active(state.family_tree)
        
        def family_info(
            next_state : BugState,
        ):
            birthdays = next_state.family_tree.player_state.players[...,0]
            current_time = next_state.family_tree.player_state.current_time
            child_locations, = jnp.nonzero(
                (birthdays == current_time) & (current_time != 0),
                size=params.max_players,
                fill_value=params.max_players,
            )
            parent_info = next_state.family_tree.parents[child_locations]
            parent_locations = parent_info[...,1]
            
            return parent_locations, child_locations
    
    return Bugs
