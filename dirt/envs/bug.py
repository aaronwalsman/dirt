from typing import TypeVar, Tuple, Any

import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.static_dataclass import static_dataclass
from mechagogue.player_list import birthday_player_list, player_family_tree

from dirt.defaults import DEFAULT_FLOAT_DTYPE
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
    consumed : Consumable
    
    # tracking
    family_tree : Any

'''
resources:
health:
    expended = somebody_bit_me
    gained = trade energy

energy is expended every step:
    expended = ambient + movement + healing + birth
    gained = eaten + photosynthesized

biomass is retained forever:
    expended = birth
    gained = eaten

water evaporates every step:
    expended = ambient + movement + healing + birth
    gained = eaten

traits:
photosynthesis:
    gain small ammount of energy from light
    but requires more biomass
size:
    requires more biomass
    but better armor and biting and photosynthesis gets you more
speed:
    able to move faster
    but requires more biomass and more energy per step
armor:
    resistant to biting
    but requires more biomass and makes you slower
teeth:
    better biting
    but requires more biomass
'''

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
        
        player_grid = jnp.zeros(params.world_size, dtype=jnp.int32)
        player_grid = player_grid.at[x[...,0], x[...,1]].set(
            jnp.arange(params.max_players))
        
        age = jnp.zeros((params.max_players,), dtype=float_dtype)
        
        health = active_players.astype(float_dtype)
        energy = active_players.astype(float_dtype)
        biomass = active_players.astype(float_dtype)
        water = active_players.astype(float_dtype)
        consumed = Consumable(energy, biomass, water)
        
        return BugState(
            x,
            r,
            player_grid,
            age,
            
            health,
            
            consumed,
            
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
        # figure out who eats what, and transfer into state
        # then return state and leftovers
        return state, consumable
    
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
        
        resources = next_state.consumed
        
        # update the energy
        next_energy = resources.energy + energy_offset
        next_energy_clipped = jnp.where(next_energy < 0., next_energy, 0.)
        
        # damage due to running out of energy
        health_offset = jnp.where(next_energy < 0., next_energy, 0.)
        next_health = next_state.health + health_offset
        
        resources = resources.replace(energy=next_energy_clipped)
        
        next_state = next_state.replace(
            consumed=resources,
            health=next_health,
        )
        
        return next_state
    
    '''
    def transition(
        key : chex.PRNGKey,
        state : TBugState,
        action : TBugAction,
        traits : TBugTraits,
    ):
        # update the traits
        state = state.update(traits=traits)
        
        # get active players
        
        # move
        
        # age
        state = state.update(age=(state.age + active_players) * active_players)
        
        # eat
    '''
        
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
