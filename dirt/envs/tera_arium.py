import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.static_dataclass import static_dataclass

@static_dataclass
class TeraAriumParams:
    world_size : Tuple[int, int] = (1024, 1024)
    
    initial_players : int = 1024
    max_players : int 16384

@static_dataclass
class TeraAriumState:
    
    # grid shaped data
    player_grid : jnp.array
    landscape : LandScapeState
    #height_grid : jnp.array
    #water_grid : jnp.array
    #ground_chemical_grid : jnp.array
    #water_chemical_grid : jnp.array
    #air_chemical_grid : jnp.array
    
    # player shaped data
    player_x : jnp.ndarray
    player_r : jnp.ndarray
    player_chemicals : jnp.ndarray

@static_dataclass
class TeraAriumTraits:
    pass

def tera_arium(params : TTeraAriumParams = TeraAriumParams()):
    
    init_players, step_players, active_players = birthday_player_list(
        params.max_players)
    init_family_tree, step_family_tree, active_family_tree = player_family_tree(
        init_players, step_players, active_players, 1)
    
    init_metabolism, step_metabolism = metabolism(params.metabolism_params)
    
    init_climate, step_climate = climate(...)
    init_hydrology, step_hydrology = hydrology(...)
    init_geology, step_geology = geology(...)
    
    def init_state(
        key : chex.PRNGKey,
        traits : TTeraAriumTraits,
    ) -> TTeraAriumState :
        
        # climate
        key, climate_key = jrng.split(key)
        climate_state = init_climate(climate_key)
        
        # hydrology
        key, hydrology_state = jrng.split(key)
        hydrology_state = init_water(hydrology_key)
        
        # geology
        key, geology_key = jrng.split(key)
        geology_state = init_geology(geology_key)
        
        # seasons
        key, season_key = jrng.split(key)
        season_state = init_season(season_key)
        
        # players
        # - initialize player tracking
        family_tree = init_family_tree(params.initial_players)
        active_players = active_family_tree(family_tree)
        
        # - initialize player age
        
        # - initialize the player positions
        key, xr_key = jrng.split(key)
        player_x, player_r = spawn.unique_xr(
            xr_key,
            params.max_players,
            params.world_size,
            active=active_players,
        )
    
    def observe(
        key : chex.PRNGKey,
        state : TTeraAriumState,
    ) -> TTeraAriumObservation:
        # player internal state
        
        # player external observation
    
    def transition(
        key : chex.PRNGKey,
        state : TTeraAriumState,
        action : TTeraAriumAction,
        traits : TTeraAriumTraits,
    ) -> TTeraAriumState :
        
        # players
        deaths = jnp.zeros(params.max_players, dtype=jnp.bool)
        
        # - eat
        #   do this before anything else happens so that the food an agent
        #   observed in the last time step is still in the right location
        pass
        
        # -- pull resources out of the environment
        pass
        
        # -- metabolize
        #    do aging and starvation here as well
        pass
        
        # - fight
        pass
        
        # - move players
        pass
        
        # - reproduce
        pass
        
        # - step family tree
        family_tree, child_locations = step_family_tree(
            state.family_tree, deaths, parents)
        
        # climate
        pass
        
        # hydrology
        pass
        
        # geology
        pass
        
        # seasons
        pass
