from typing import TypeVar, Tuple, Any

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.static_dataclass import static_dataclass
from mechagogue.player_list import birthday_player_list, player_family_tree

from dirt.constants import (
    DEFAULT_FLOAT_DTYPE, DEFAULT_BUG_COLOR, PHOTOSYNTHESIS_COLOR)
import dirt.gridworld2d.dynamics as dynamics
import dirt.gridworld2d.spawn as spawn
from dirt.consumable import Consumable

TBugParams = TypeVar('TBugParams', bound='BugParams')
TBugState = TypeVar('TBugState', bound='BugState')
TBugAction = TypeVar('TBugAction', bound='BugAction')
TBugTraits = TypeVar('TBugTrait', bound='BugTraits')
TBugObservation = TypeVar('TBugObservation', bound='BugObservation')

@static_dataclass
class BugParams:
    world_size : Tuple[int,int] = (1024,1024)
    initial_players : int = 1024
    max_players : int = 16384
    
    base_size_metabolism : float = 0.01
    base_brain_metabolism : float = 0.01
    base_move_metabolism : float = 0.01
    base_climb_metabolism : float = 0.01

@static_dataclass
class BugTraits:
    body_size : jnp.ndarray
    brain_size : jnp.ndarray
    
    base_color : jnp.ndarray
       
    photosynthesis : jnp.ndarray
    
    #speed : jnp.ndarray
    #armor : jnp.ndarray
    #teeth : jnp.ndarray

@static_dataclass
class BugAction:
    forward : jnp.ndarray
    rotate : jnp.ndarray
    bite : jnp.ndarray
    eat : jnp.ndarray
    reproduce : jnp.ndarray

@static_dataclass
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

def bugs(
    params : TBugParams = BugParams(),
    float_dtype : Any = DEFAULT_FLOAT_DTYPE
):
    
    init_players, step_players, active_players = birthday_player_list(
        params.max_players)
    init_family_tree, step_family_tree, active_family_tree = player_family_tree(
        init_players, step_players, active_players, 1)
    
    def init_state(
        key : chex.PRNGKey,
    ) -> TBugState :
        
        family_tree = init_family_tree(params.initial_players)
        active_players = active_family_tree(family_tree)
        
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
        
        age = jnp.zeros((params.max_players,), dtype=jnp.int32)
        
        health = active_players.astype(float_dtype)
        energy = active_players.astype(float_dtype)
        biomass = active_players.astype(float_dtype)
        water = active_players.astype(float_dtype)
        stomach = Consumable(energy, biomass, water)
        
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
            
            family_tree,
        )
    
    def move(
        state : TBugState,
        action : TBugAction,
    ):
        
        active_bugs = active_family_tree(state.family_tree)
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
        state : TBugState,
        consumable : Consumable,
        action : TBugAction,
    ):
        live_eaters = active_players(state) * action.eat
        
        # figure out who eats what, and transfer into state
        # then return state and leftovers
        max_stomach_water = 1.
        drunk_water = jnp.clip(
            consumable.water * live_eaters,
            max=max_stomach_water-state.stomach.water,
        )
        eaten_energy = consumable.energy * live_eaters
        eaten_biomass = consumable.biomass * live_eaters
        
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
        state : TBugState,
        next_state : TBugState,
        traits : TBugTraits,
        terrain : jnp.ndarray,
        water : jnp.ndarray,
    ):
        energy_offset = jnp.zeros(params.max_players, dtype=float_dtype)
        
        volume = traits.body_size**3
        
        # size tax
        energy_offset -= params.base_size_metabolism * volume
        
        # brain tax
        energy_offset -= params.base_brain_metabolism * traits.brain_size
        
        # movement
        x0 = state.x
        r0 = state.r
        x1 = next_state.x
        r1 = next_state.r
        dx = jnp.abs(x1-x0).sum(axis=-1)
        dr = dynamics.distance_r(r1, r0)
        energy_offset -= params.base_move_metabolism * volume * (dx + dr)
        
        # climbing
        height = terrain + water
        dy = height[x1[...,0], x1[...,1]] - height[x0[...,0], x0[...,1]]
        uphill = jnp.where(dy > 0., dy, 0.)
        energy_offset -= uphill * volume * params.base_climb_metabolism
        
        next_stomach = next_state.stomach
        
        # update the energy
        next_energy = next_stomach.energy + energy_offset
        next_energy_clipped = jnp.where(next_energy < 0., next_energy, 0.)
        
        # damage due to running out of energy
        health_offset = jnp.where(next_energy < 0., next_energy, 0.)
        next_health = next_state.health + health_offset
        
        next_stomach = next_stomach.replace(energy=next_energy_clipped)
        
        # color
        color = (
            traits.base_color * (1. - traits.photosynthesis[:,None]) + 
            PHOTOSYNTHESIS_COLOR * (traits.photosynthesis[:,None])
        )
        
        # deaths
        # - figure out who died
        deaths = next_health <= 0.
        # - update the next state variables appropriately
        next_health = next_health * ~deaths
        
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
        next_family_tree, children = step_family_tree(
            next_state.family_tree,
            deaths,
            jnp.full((n, 1), -1, dtype=jnp.int32),
        )
        
        next_state = next_state.replace(
            x=x2,
            stomach=next_stomach,
            health=next_health,
            color=color,
            family_tree=next_family_tree,
        )
        
        return next_state, expelled, expelled_locations
        
    def active_players(
        state : TBugState,
    ):
        return active_family_tree(state.family_tree)
    
    def family_info(
        next_state : TBugState,
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
    
    return init_state, move, eat, metabolism, active_players, family_info
