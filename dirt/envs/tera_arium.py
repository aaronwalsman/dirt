from typing import TypeVar, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.static_dataclass import static_dataclass

from mechagogue.dp.population_game import population_game

from dirt.constants import (
    ROCK_COLOR,
    WATER_COLOR,
    ENERGY_TINT,
    BIOMASS_TINT,
)
from dirt.envs.landscape import (
    LandscapeParams,
    LandscapeState,
    landscape,
)
from dirt.envs.bug import (
    BugParams,
    BugAction,
    BugTraits,
    BugState,
    bugs,
)

TTeraAriumParams = TypeVar('TTeraAriumParams', bound='TeraAriumParams')
TTeraAriumState = TypeVar('TTeraAriumState', bound='TeraAriumState')
TTeraAriumObservation = TypeVar(
    'TTeraAriumObservation', bound='TeraAriumObservation')

@static_dataclass
class TeraAriumParams:
    world_size : Tuple[int, int] = (1024, 1024)
    
    initial_players : int = 1024
    max_players : int = 16384
    
    landscape : LandscapeParams = LandscapeParams()
    bugs : BugParams = BugParams()

@static_dataclass
class TeraAriumState:
    
    landscape : LandscapeState
    #height_grid : jnp.array
    #water_grid : jnp.array
    #ground_chemical_grid : jnp.array
    #water_chemical_grid : jnp.array
    #air_chemical_grid : jnp.array
    
    # player shaped data
    bugs : BugState
    #player_x : jnp.ndarray
    #player_r : jnp.ndarray
    #player_chemicals : jnp.ndarray

@static_dataclass
class TeraAriumObservation:
    rgb : jnp.ndarray
    height : jnp.ndarray
    
    ground_water : jnp.ndarray
    ground_energy : jnp.ndarray
    ground_biomass : jnp.ndarray

@static_dataclass
class TeraAriumTraits:
    pass

TeraAriumAction = BugAction
TeraAriumTraits = BugTraits

def render_tera_arium(
    water,
    energy,
    biomass,
    bug_x,
    bug_color,
    light,
):
    h, w = water.shape

    # start with a baseline rock color of 50% gray
    rgb = jnp.full((h,w,3), ROCK_COLOR, dtype=water.dtype)

    # overlay the water as blue
    rgb = jnp.where(water[..., None] > 0.05, WATER_COLOR, rgb)

    # apply the energy tint
    clipped_energy = jnp.clip(energy, min=0., max=1.)
    rgb = rgb + clipped_energy[..., None] * ENERGY_TINT

    # apply the biomass tint
    clipped_biomass = jnp.clip(biomass, min=0., max=1.)
    rgb = rgb + clipped_biomass[..., None] * BIOMASS_TINT
    
    # overlay the bug colors
    rgb = rgb.at[bug_x[...,0], bug_x[...,1]].set(bug_color)
    
    # apply lighting
    rgb = rgb * light[..., None]
    
    # clip between 0 and 1
    rgb = jnp.clip(rgb, min=0., max=1.)
    
    return rgb

def tera_arium(params : TTeraAriumParams = TeraAriumParams()):
    
    #init_players, step_players, active_players = birthday_player_list(
    #    params.max_players)
    #init_family_tree, step_family_tree, active_family_tree = player_family_tree(
    #    init_players, step_players, active_players, 1)
    
    #init_metabolism, step_metabolism = metabolism(params.metabolism_params)
    
    #init_climate, step_climate = climate(...)
    #init_hydrology, step_hydrology = hydrology(...)
    #init_geology, step_geology = geology(...)
    
    #landscape_params = params.landscape.replace(
    #    world_size=params.world_size,
    #)
    
    (
        init_landscape,
        get_landscape_consumable,
        set_landscape_consumable,
        add_landscape_consumable,
        step_landscape,
    ) = landscape(params.landscape)
    #bug_params = params.bugs.replace(
    #    initial_players=params.initial_players,
    #    max_players=params.max_players,
    #)
    (
        init_bugs,
        move_bugs,
        bugs_eat,
        bug_metabolism,
        active_bugs,
        bug_family_info,
    ) = bugs(params.bugs)
    
    def init_state(
        key : chex.PRNGKey,
    ) -> TTeraAriumState :
        
        key, landscape_key = jrng.split(key)
        landscape_state = init_landscape(landscape_key)
        
        key, bug_key = jrng.split(key)
        bug_state = init_bugs(bug_key)
        
        state = TeraAriumState(landscape_state, bug_state)
        
        return state
    
    def observe(
        key : chex.PRNGKey,
        state : TTeraAriumState,
    ) -> TTeraAriumObservation:
        # player internal state
        
        # player external observation
        return None
    
    def transition(
        key : chex.PRNGKey,
        state : TTeraAriumState,
        action : BugAction,
        traits : BugTraits,
    ) -> TTeraAriumState :
        
        next_state = state
        
        # bugs
        bug_state = next_state.bugs
        
        max_players = state.bugs.family_tree.parents.shape[0]
        deaths = jnp.zeros(max_players, dtype=jnp.bool)
        
        # - eat
        #   do this before anything else happens so that the food an agent
        #   observed in the last time step is still in the right location
        # -- pull consumables out of the environment
        next_landscape_state = next_state.landscape
        landscape_consumable = get_landscape_consumable(
            next_landscape_state, bug_state.x)
        # -- eat
        next_bug_state, leftovers = bugs_eat(
            bug_state, landscape_consumable, action)
        # -- put the leftovers back into the landscape
        next_landscape_state = set_landscape_consumable(
            next_landscape_state, bug_state.x, leftovers)
        
        # - fight
        pass
        
        # - move bugs
        next_bug_state = move_bugs(next_bug_state, action)
        
        # - metabolize and reproduce
        next_bug_state, expelled, expelled_locations = bug_metabolism(
            bug_state,
            action,
            next_bug_state,
            traits,
            next_landscape_state.terrain,
            next_landscape_state.water,
            next_landscape_state.air_light,
        )
        # -- put the expelled resources back into the landscape
        next_landscape_state = add_landscape_consumable(
            next_landscape_state, expelled_locations, expelled)
        
        # natural landscape processes
        key, landscape_key = jrng.split(key)
        next_landscape_state = step_landscape(
            landscape_key, next_landscape_state)
        
        next_state = next_state.replace(
            landscape=next_landscape_state,
            bugs=next_bug_state
        )
        
        return next_state
    
    def active_players(state):
        return active_bugs(state.bugs)
    
    def family_info(state):
        return bug_family_info(state.bugs)
    
    return population_game(
        init_state, transition, observe, active_players, family_info)
