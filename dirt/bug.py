from typing import TypeVar, Tuple, Any

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
    have to clip if we were doing this in a constraint-based setup.
    
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
MOVE_ACTION_TYPE = 0
ATTACK_ACTION_TYPE = 1
EAT_ACTION_TYPE = 2
EXPELL_ACTION_TYPE = 3
SMELL_ACTION_TYPE = 4
AUDIO_ACTION_TYPE = 5
LEVEL_ACTION_TYPE = 6
REPRODUCE_ACTION_TYPE = 7
NUM_ACTION_TYPES = 8

@static_data
class BugParams:
    world_size : Tuple[int,int] = (1024,1024)
    #initial_players : int = 1024
    max_players : int = 16384
    
    include_water : bool = True
    include_energy : bool = True
    include_biomass : bool = True
    
    levels : int = 0
    
    zero_water_damage : float = 0.1
    
    # energy costs
    body_size_energy_cost : float = 0.01
    brain_size_energy_cost : float = 0.01
    move_energy_cost : float = 0.01
    climb_energy_cost : float = 0.01
    birth_energy_cost : float : 0.01
    
    # biomass requirements
    # TODO
    
    photosynthesis_energy_gain : float = 0.005

@static_data
class BugTraits:
    # brain
    brain_size : jnp.ndarray
    
    # color
    target_color : jnp.ndarray
    
    # resources
    photosynthesis : jnp.ndarray
    
    # sensing
    visual_range : jnp.ndarray
    visual_sensitivity : jnp.ndarray
    audio_sensitivity : jnp.ndarray
    smell_sensitivity : jnp.ndarray
    internal_sensitivity : jnp.ndarray
    
    # efficiency
    max_climb : jnp.ndarray
    climb_efficiency : jnp.ndarray
    move_effciency : jnp.ndarray
    
    # stomach
    max_water : jnp.ndarray
    water_gulp : jnp.ndarray
    max_energy : jnp.ndarray
    energy_gulp : jnp.ndarray
    max_biomass : jnp.ndarray
    biomass_gulp : jnp.ndarray
    
    # climate
    insulation : jnp.ndarray
    
    # armor
    armor : jnp.ndarray
    
    # age
    senescence : jnp.ndarray
    
    # health
    healing_rate : jnp.ndarray
    
    # reproduction
    newborn_energy : jnp.ndarray
    newborn_biomass : jnp.ndarray
    newborn_water : jnp.ndarray
    newborn_color : jnp.ndarray
    
    # actions
    movement_primitives : jnp.ndarray
    attack_primitives : jnp.ndarray

'''
@static_data
class BugAction:
    forward : jnp.ndarray
    rotate : jnp.ndarray
    bite : jnp.ndarray
    eat : jnp.ndarray
    reproduce : jnp.ndarray

Kinds of Actions:
    Movement
    Attack
    Eat
    Expell
    Scent-mark
    Call
    Change Level
    Reproduce
'''

@static_data
class BugAction:
    move : jnp.ndarray
    bite : jnp.ndarray
    eat : jnp.ndarray
    reproduce : jnp.ndarray

@static_data
class BugState:
    # location
    x : jnp.ndarray
    r : jnp.ndarray
    object_grid : jnp.ndarray
    
    # age
    age : jnp.ndarray
    
    # resources
    health : jnp.ndarray
    water : jnp.ndarray
    energy : jnp.ndarray
    biomass : jnp.ndarray
    
    # color
    color : jnp.ndarray
    
    # level
    level : jnp.ndarray
    
    # tracking
    family_tree : Any

def make_bugs(
    params : BugParams = BugParams(),
    float_dtype : Any = DEFAULT_FLOAT_DTYPE
):
    
    player_list = birthday_player_list(params.max_players)
    family_tree = player_family_tree(player_list, 1)
    
    # construct the action mapping
    # - figure out how many eat and expell action primitives there are
    eat_primitives = (
        int(params.include_water) +
        int(params.include_energy) +
        int(params.include_biomass)
    expell_primitves = eat_primitives
    
    # - compute the total number of actions
    num_actions = (
        params.movement_primitives +
        params.attack_primitives +
        eat_primitives +
        expell_primitives +
        params.smell_primitives +
        params.audio_primitives +
        params.levels +
        1
    )
    
    # - build the mapping that goes from integers to action type and primitive
    #   index, and vice-versa
    action_to_primitive_map = jnp.zeros((num_actions, 2), dtype=jnp.int32)
    primitive_to_action_map = jnp.zeros(
        (NUM_ACTION_TYPES, MAX_ACTION_PRIMITIVES), dtype=jnp.int32)
    action_index = 0
    for action_type, actions_per_type in (
        (MOVE_ACTION_TYPE, params.movement_primitives),
        (ATTACK_ACTION_TYPE, params.attack_primitives),
        (EAT_ACTION_TYPE, eat_primitives),
        (EXPELL_ACTION_TYPE, expell_primitives),
        (SMELL_ACTION_TYPE, params.smell_primitives),
        (AUDIO_ACTION_TYPE, params.audio_primitives),
        (LEVEL_ACTION_TYPE, params.levels),
        (REPRODUCE_ACTION_TYPE, 1),
    ):
        for j in range(actions_per_type):
            action_to_primitive_map[action_index] = (action_type, j)
            primitive_to_action_map[action_type, j] = action_index
            action_index += 1
    
    @static_functions
    class Bugs:
        def init(
            key : chex.PRNGKey,
            initial_players,
            parent_traits : BugTraits,
            x : Optional[jnp.ndarray] = None,
            r : Optional[jnp.ndarray] = None,
        ) -> BugState :
            
            assert initial_players <= params.max_players, (
                f'initial_players ({initial_players}) must be less than or'
                f'equal to params.max_players ({max_players})'
            )
            
            # initialize the family tree
            family_tree_state = family_tree.init(initial_players)
            active_players = family_tree.active(family_tree_state)
            
            # initialize map positions and rotations
            assert (x is None) == (r is None), (
                'must specify either both x and r or neither')
            if x is None:
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
            
            # when using levels, select the first level of the parent_traits
            if params.levels:
                parent_traits = tree_getitem(parent_traits, 0)
            
            # initialize health, water, energy, and biomass
            health = active_players.astype(float_dtype)
            water = (
                active_players*parent_traits.newborn_water).astype(float_dtype)
            energy = (
                active_players*parent_traits.newborn_energy).astype(float_dtype)
            biomass = (
                active_players*parent_traits.newborn_health).astype(float_dtype)
            
            # initialize color
            n = active_players.shape[0]
            color = jnp.full((n, 3), parent_traits.newborn_color)
            
            return BugState(
                x,
                r,
                object_grid,
                age,
                
                health,
                energy,
                biomass,
                water,
                
                color,
                
                family_tree_state,
            )
        
        def move(
            state : BugState,
            action : BugAction,
            traits : BugTraits,
        ):
            
            active_bugs = family_tree.active(state.family_tree)
            x, r, _, object_grid = dynamics.movement_step(
                state.x,
                state.r,
                action.move,
                active=active_bugs,
                check_collisions=True,
                object_grid=state.object_grid,
            )
            state = state.replace(x=x, r=r, object_grid=object_grid)
            
            return state
        
        '''
        def eat(
            state : BugState,
            consumable : Consumable,
            action : BugAction,
        ):
            live_eaters = active_players(state) * action.eat
            
            # figure out who eats what, and transfer into state
            # then return state and leftovers
            max_water = jnp.clip(
                params.max_stomach_water - state.stomach.water,
                max=params.max_water_gulp,
            )
            drunk_water = jnp.clip(
                consumable.water * live_eaters,
                max=max_water,
            )
            
            eaten_energy = consumable.energy * live_eaters
            max_biomass = jnp.clip(
                params.max_stomach_biomass - state.stomach.biomass,
                max=params.max_biomass_gulp,
            )
            eaten_biomass = jnp.clip(
                consumable.biomass * live_eaters,
                max=max_biomass,
            )
            
            next_stomach = state.stomach.replace(
                water = state.stomach.water + drunk_water,
                energy = state.stomach.energy + eaten_energy,
                biomass = state.stomach.biomass + eaten_biomass,
            )
            next_state = state.replace(stomach=next_stomach)
            
            leftovers = consumable.replace(
                water=consumable.water - drunk_water,
                energy=consumable.energy - eaten_energy,
                biomass=consumable.biomass - eaten_biomass,
            )
            
            return next_state, leftovers
        '''
        
        def metabolism(
            state : BugState,
            action : BugAction,
            next_state : BugState,
            traits : BugTraits,
            terrain : jnp.ndarray,
            water : jnp.ndarray,
            light : jnp.ndarray,
        ):
            
            next_stomach = next_state.stomach
            volume = traits.body_size**3
            alive = active_players(next_state)
            
            # compute energy expenditure
            energy_offset = jnp.zeros(params.max_players, dtype=float_dtype)
            water_offset = jnp.zeros(params.max_players, dtype=float_dtype)
            biomass_offset = jnp.zeros(params.max_players, dtype=float_dtype)
            health_offset = jnp.zeros(params.max_players, dtype=float_dtype)
            
            # - healing
            #   add health if less than 1 at the expense of energy
            heal_ammount = jnp.clip(
                (1. - next_state.health), min=0, max=params.healing_rate)
            health_offset += heal_ammount
            energy_offset -= heal_ammount
            
            # - size energy tax
            energy_offset -= params.base_size_metabolism * volume
            
            # - brain energy tax
            energy_offset -= params.base_brain_metabolism * traits.brain_size
            
            # - movement energy tax
            x0 = state.x
            r0 = state.r
            x1 = next_state.x
            r1 = next_state.r
            dx = jnp.abs(x1-x0).sum(axis=-1)
            dr = dynamics.distance_r(r1, r0)
            energy_offset -= params.base_move_metabolism * volume * (dx + dr)
            
            # - climbing energy tax
            height = terrain + water
            dy = height[x1[...,0], x1[...,1]] - height[x0[...,0], x0[...,1]]
            uphill = jnp.where(dy > 0., dy, 0.)
            energy_offset -= uphill * volume * params.base_climb_metabolism
            
            # - photosynthesis
            energy_offset += (
                traits.photosynthesis *
                params.photosynthesis_energy_gain *
                light[x0[...,0], x0[...,1]]
            )
            
            # - compute the updated energy
            #   (do no more energy changes after this point)
            next_energy = next_stomach.energy + energy_offset
            next_energy_clipped = jnp.where(next_energy < 0., 0., next_energy)
            
            # - compute the updated water
            #   (no no more water changes after this point)
            next_water = next_stomach.water + water_offset
            next_water_clipped = jnp.where(next_water < 0., 0., next_water)
            
            # - damage due to using more energy than is available
            health_offset += jnp.where(next_energy < 0., next_energy, 0.)
            
            # - damage due to using more water than is available
            health_offset += jnp.where(next_water < 0., next_water, 0.)
            
            # - damage due to having zero water
            health_offset += jnp.where(
                next_water <= 0., -params.zero_water_damage, 0)
            
            # reproduce
            # - compute which agents are able to reproduce
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
            birthdays = next_state.family_tree.player_list.players[...,0]
            current_time = next_state.family_tree.player_list.current_time
            child_locations, = jnp.nonzero(
                birthdays == current_time,
                size=params.max_players,
                fill_value=params.max_players,
            )
            parent_info = next_state.family_tree.parents[child_locations]
            parent_locations = parent_info[...,1]
            
            return parent_locations, child_locations
    
    return Bugs
