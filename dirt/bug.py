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
from dirt.consumable import Consumable

@static_data
class BugParams:
    world_size : Tuple[int,int] = (1024,1024)
    initial_players : int = 1024
    max_players : int = 16384
    
    # starting characteristics (make traits?)
    starting_health : float = 0.5
    starting_energy : float = 1.
    starting_water : float = 0.1
    starting_biomass : float = 1.
    
    max_stomach_water : float = 1.
    max_water_gulp : float = 0.1
    max_stomach_biomass : float = 2.
    max_biomass_gulp : float = 0.1
    
    healing_rate : float = 0.1
    zero_water_damage : float = 0.1
    
    birth_energy : float = 0.25
    birth_damage : float = 0.25
    birth_water : float = 0.25
    
    base_size_metabolism : float = 0.01
    base_brain_metabolism : float = 0.01
    base_move_metabolism : float = 0.01
    base_climb_metabolism : float = 0.01
    
    senescence : float = 0.0 #0.001
    
    photosynthesis_energy_gain : float = 0.005

@static_data
class BugTraits:
    body_size : jnp.ndarray
    brain_size : jnp.ndarray
    
    base_color : jnp.ndarray
       
    photosynthesis : jnp.ndarray
    
    movement_primitives : jnp.ndarray

@static_data
class BugAction:
    forward : jnp.ndarray
    rotate : jnp.ndarray
    bite : jnp.ndarray
    eat : jnp.ndarray
    reproduce : jnp.ndarray

@static_data
class BugState:
    x : jnp.ndarray
    r : jnp.ndarray
    object_grid : jnp.ndarray
    age : jnp.ndarray
    
    # resources
    health : jnp.ndarray
    stomach : Consumable
    
    # color
    color : jnp.ndarray
    
    # tracking
    family_tree : Any

def make_bugs(
    params : BugParams = BugParams(),
    float_dtype : Any = DEFAULT_FLOAT_DTYPE
):
    
    player_list = birthday_player_list(params.max_players)
    family_tree = player_family_tree(player_list, 1)
    
    @static_functions
    class Bugs:
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
            
            # initialize health, energy, biomass, water and stomach
            health = (
                active_players * params.starting_health).astype(float_dtype)
            energy = (
                active_players * params.starting_energy).astype(float_dtype)
            biomass = (
                active_players * params.starting_health).astype(float_dtype)
            water = (
                active_players * params.starting_water).astype(float_dtype)
            stomach = Consumable(energy, biomass, water)
            
            # initialize color
            n = active_players.shape[0]
            color = jnp.full((n, 3), DEFAULT_BUG_COLOR)
            
            return BugState(
                x,
                r,
                object_grid,
                age,
                
                health,
                stomach,
                
                color,
                
                family_tree_state,
            )
        
        def move(
            state : BugState,
            action : BugAction,
        ):
            
            active_bugs = family_tree.active(state.family_tree)
            x, r, _, object_grid = dynamics.forward_rotate_step(
                state.x,
                state.r,
                action.forward,
                action.rotate,
                active=active_bugs,
                check_collisions=True,
                object_grid=state.object_grid,
            )
            state = state.replace(x=x, r=r, object_grid=object_grid)
            
            return state
        
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
