from typing import TypeVar, Tuple, Any, Optional

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

@static_data
class BugParams:
    world_size : Tuple[int,int] = (1024,1024)
    initial_players : int = 1024
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
    birth_energy_cost : float = 0.01
    color_change_energy_cost : float = 0.01
    
    # biomass requirements
    # TODO
    
    # mass
    biomass_mass : float = 1.
    water_mass : float = 1.
    
    photosynthesis_energy_gain : float = 0.005
    
    # actions
    movement_primitives : int = 3
    attack_primitives : int = 1
    odor_primitives : int = 1
    audio_primitives : int = 1
    
    # initialization
    initial_health : float = 1.
    initial_water : float = 0.1
    initial_energy : float = 1.
    initial_biomass : float = 1.0
    initial_color : Tuple[float, float, float] = (0.5, 0.25, 0.5)

    def validate(params):
        assert params.initial_players <= params.max_players, (
            f'params.initial_players ({params.initial_players}) must be less '
            f'than or equal to params.max_players ({max_players})'
        )
        
        return params

@static_data
class BugTraits:
    # brain
    brain_size : float | jnp.ndarray = 0.
    
    # body
    #body_size : jnp.ndarray
    
    # color
    color : Tuple[float, float, float] | jnp.ndarray = DEFAULT_BUG_COLOR
    
    # resources
    photosynthesis : float | jnp.ndarray = 0.
    
    # sensing
    view_distance : int | jnp.ndarray = 5
    view_back_distance : int | jnp.ndarray = 0
    view_width : int | jnp.ndarray = 5
    view_sensitivity : float | jnp.ndarray = 1.
    audio_sensitivity : float | jnp.ndarray = 1.
    smell_sensitivity : float | jnp.ndarray = 1.
    internal_sensitivity : float | jnp.ndarray = 1.
    
    # efficiency
    max_climb : float | jnp.ndarray = 2.
    climb_efficiency : float | jnp.ndarray = 1.
    move_effciency : float | jnp.ndarray = 1.
    
    # stomach
    max_water : float | jnp.ndarray = 1.
    water_gulp : float | jnp.ndarray = 0.5
    max_energy : float | jnp.ndarray = 2.
    energy_gulp : float | jnp.ndarray = 0.5
    max_biomass : float | jnp.ndarray = 5.
    biomass_gulp : float | jnp.ndarray = 0.5
    
    # climate
    insulation : float | jnp.ndarray = 0.
    
    # armor
    armor : float | jnp.ndarray = 0.
    
    # age
    senescence : float | jnp.ndarray = 0.999
    
    # health
    healing_rate : float | jnp.ndarray = 0.1
    
    # reproduction
    newborn_energy : float | jnp.ndarray = 1.0
    newborn_biomass : float | jnp.ndarray = 1.0
    newborn_water : float | jnp.ndarray = 0.5
    newborn_color : Tuple[float, float, float] | jnp.ndarray = DEFAULT_BUG_COLOR
    
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
    
    # eat primitives
    eat_primitive = 0
    if params.include_water:
        DRINK_WATER_PRIMITIVE = eat_primitive
        eat_primitive += 1
    if params.include_energy:
        EAT_ENERGY_PRIMITIVE = eat_primitive
        eat_primitive += 1
    if params.include_biomass:
        EAT_BIOMASS_PRIMITIVE = eat_primitive
        eat_primitive += 1
    
    @static_functions
    class Bugs:
        num_actions = num_action_primitives
        
        def init(
            key : chex.PRNGKey,
            #initial_players,
            #parent_traits : BugTraits,
            #x : Optional[jnp.ndarray] = None,
            #r : Optional[jnp.ndarray] = None,
        ) -> BugState :
            
            # initialize the family tree
            family_tree_state = family_tree.init(params.initial_players)
            active_players = family_tree.active(family_tree_state)
            
            # initialize map positions and rotations
            #assert (x is None) == (r is None), (
            #    'must specify either both x and r or neither')
            #if x is None:
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
            #if params.levels:
            #    parent_traits = tree_getitem(parent_traits, 0)
            
            # initialize health, water, energy, and biomass
            health = (
                active_players*params.initial_health).astype(float_dtype)
            water = (
                active_players*params.initial_water).astype(float_dtype)
            energy = (
                active_players*params.initial_energy).astype(float_dtype)
            biomass = (
                active_players*params.initial_biomass).astype(float_dtype)
            
            # initialize color
            n = active_players.shape[0]
            color = jnp.full(
                (n, 3), jnp.array(params.initial_color, dtype=float_dtype))
            
            # initialize level
            level = jnp.zeros(params.max_players, dtype=jnp.int32)
            
            return BugState(
                x=x,
                r=r,
                object_grid=object_grid,
                age=age,
                
                health=health,
                energy=energy,
                biomass=biomass,
                water=water,
                
                color=color,
                
                level=level,
                
                family_tree=family_tree_state,
            )
        
        def move(
            state : BugState,
            action : int,
            traits : BugTraits,
        ):
            
            active_bugs = family_tree.active(state.family_tree)
            x, r, _, object_grid = dynamics.forward_rotate_step(
                state.x,
                state.r,
                #action.forward,
                #action.rotate,
                # REPLACE WITH ACTION PRIMITIVES
                active=active_bugs,
                check_collisions=True,
                object_grid=state.object_grid,
            )
            state = state.replace(x=x, r=r, object_grid=object_grid)
            
            return state
        
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
            eat = action_type == EAT_ACTION_TYPE
            
            # water
            if params.include_water:
                # - compute which bugs are drinking water
                drink_water = eat & (action_primitive == DRINK_WATER_PRIMITIVE)
                # - compute how much water each bug will consume
                desired_water = drink_water * jnp.minimum(
                    traits.water_gulp, traits.max_water - state.water)
                consumed_water = jnp.minimum(desired_water, water)
                # - compute the lefover water and the bugs' new water
                leftover_water = water - consumed_water
                new_water = state.water + consumed_water
                # - update state
                state = state.replace(water=new_water)
            else:
                leftover_water = None
            
            # energy
            if params.include_energy:
                # - compute which bugs are eating energy
                eat_energy = eat & (action_primitive == EAT_ENERGY_PRIMITIVE)
                # - compute how much energy each bug will consume
                desired_energy = eat_energy * jnp.minimum(
                    traits.energy_gulp, traits.max_energy - state.energy)
                consumed_energy = jnp.min(desired_energy, energy)
                # - compute the leftover energy and the bugs' new energy
                leftover_energy = energy - consumed_energy
                new_energy = state.energy + consumed_energy
                # - update state
                state = state.replace(energy=new_energy)
            else:
                leftover_energy = None
            
            # biomass
            if params.include_biomass:
                # - compute which bugs are eating biomass
                eat_biomass = eat & (action_primitive == EAT_BIOMASS_PRIMITIVE)
                # - compute how much biomass each bug will consume
                desired_biomass = eat_biomass * jnp.minimum(
                    traits.biomass_gulp, traits.max_biomass - state.biomass)
                consumed_biomass = jnp.min(desired_biomass, biomass)
                # - compute the leftover biomass and the bugs' new energy
                leftover_biomass = biomass - consumed_biomass
                new_biomass = state.biomass + consumed_biomass
                # - update state
                state = state.replace(biomass=new_biomass)
            else:
                leftover_biomass = None
            
            return state, leftover_water, leftover_energy, leftover_biomass
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
            action : int,
            next_state : BugState,
            traits : BugTraits,
            terrain : jnp.ndarray,
            water : jnp.ndarray,
            light : jnp.ndarray,
        ):
        
            mass = (
                biomass * params.biomass_mass +
                state.water * params.water_mass
            )
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
            energy_offset -= params.base_size_metabolism * mass
            
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
