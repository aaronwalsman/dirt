import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from dirt.defaults import DEFAULT_FLOAT_TYPE
from dirt.gridworld2d.gas import step as gas_step

@static_dataclass
class LandscapeParams:
    world_size : Tuple[int, int] = (1024, 1024)
    step_sizes : Tuple[float, ...] = (1.)
    
    # terrain
    terrain_octaves : int = 12
    terrain_max_octaves : Optional[int] = None
    terrain_lacunarity : float = 2.
    terrain_persistence : float = 0.5
    terrain_unit_scale : float = 0.05
    terrain_max_height : float = 50.
    
    # water
    water_per_cell : float = 2.
    water_initial_fill_rate : float = 0.01
    water_flow_rate : float = 0.01
    rain_moisture_threshold = float = 0.1
    air_moisture_diffusion = float 1./3.
    
    # air
    wind_std : float = 0.1
    wind_reversion : float = 0.001
    air_initial_temperature : float = 0.
    
    # climate
    steps_per_day : 24*10
    days_per_year : 360
    evaporation_rate : float = 0.01
    
    # erosion
    erosion_endurance : float = SOMETHING
    erosion_ratio : float = SOMETHING

@static_dataclass
class LandscapeState:
    terrain : jnp.array
    erosion : jnp.array
    water : jnp.array
    wind_velocity : jnp.array
    air_temperature : jnp.array
    air_moisture : jnp.array
    #ground_chemicals : jnp.array
    #water_chemicals : jnp.array
    #air_chemicals : jnp.array

class LandscapeAction:
    #step_size : float
    #locations : jnp.array
    #ground_chemical_update : jnp.array
    #water_chemical_update : jnp.array
    #air_chemical_update : jnp.array

LandscapeObservation = LandscapeState

def landscape(
    params : TLandscapeParams = LandscapeParams(),
    dtype : Any = DEFAULT_FLOAT_TYPE,
):
    
    init_wind_velocity, step_wind_velocity = ou_process(
        params.wind_std,
        params.wind_reversion,
        jnp.zeros((2,), dtype=dtype),
    )
    
    def init(
        key : chex.PRNGKey,
    ) -> TLandscapeState :
        
        # terrain
        # - use fractal_noise to generate an initial terrain grid
        key, terrain_key = jrng.split(key)
        terrain = fractal_noise(
            ground_key,
            params.world_size,
            params.terrain_octaves,
            params.terrain_max_octaves,
            params.terrain_lacunarity,
            params.terrain_persistence,
            params.terrain_unit_scale,
            params.terrain_max_height,
            dtype=dtype,
        )
        
        # erosion
        # - initialize erosion to be zero everywhere
        erosion = jnp.zeros(params.world_size, dtype=dtype)
        
        # water
        # - start with zero water everywhere
        water = jnp.zeros(params.world_size, dtype=dtype)
        
        # - compute the total water we want distributed over the entire grid
        total_water = (
            params.water_per_cell * params.world_size[0] * params.world_size[1])
        
        # - then we will fill the lowest points on the grid with water until
        #   the desired total water has been deposited in the grid
        water_level = jnp.min(terrain)
        
        def water_level_cond(water_water_level):
            water, water_level = water_water_level
            return jnp.sum(water) < total_water
        
        def water_level_body(water_water_level):
            water, water_level = water_water_level
            water_level += water_initial_fill_rate
            water = water_level - terrain
            water = jnp.where(water < 0., 0., water)
            return (water, water_level)
        
        water, water_level = jax.lax.while_loop(
            water_level_cond, water_level_body, (water, water_level))
        
        # - scale the water everywhere in order to counteract any overshoot
        current_water = jnp.sum(water)
        water = water * (total_water / current_water)
        
        # air
        key, wind_key = jrng.split(key)
        wind_velocity = init_wind_velocity(wind_key)
        air_temperature = jnp.full(
            params.world_size, params.initial_air_temperature, dtype=dtype)
        air_moisture = jnp.zeros(params.world_size, dtype=dtype)
        
        return LandscapeState(
            terrain,
            erosion,
            water,
            wind_velocity,
            air_temperature,
            air_moisture,
        )
    
    step_functions = []
    for step_size in step_sizes:
        def step(
            key : chex.PRNGKey,
            action : TLandscapeAction,
            state : TLandscapeState,
        ) -> Tuple[TLandscapeState, TLandscapeObservation] :
            
            terrain = state.terrain
            water = state.water
            wind_velocity = state.wind_velocity
            air_moisture = state.air_moisture
            
            # apply actions
            
            # move air
            # - update the wind direction
            #   TODO: iterate if step_size is too large
            key, wind_key = jrng.split(key)
            wind_velocity = step_wind_velocity(
                wind_key, step_size=step_size)
            
            # - diffuse and move the air moisture
            diffusion_std = params.air_moisture_diffusion * (step_size**0.5)
            air_moisture = gas_step(
                air_moisture[...,None], diffusion_std, 1., wind_velocity, 1)
            
            # move water
            # - evaporate
            #   TODO: incorporate temperature?
            evaporation = jnp.minimum(
                water, params.evaporation_rate * action.step_size)
            water = water - evaporation
            air_moisture = air_moisture + evaporation
            
            # - rain
            #   TODO: incorporate temperature?
            rain = air_moisture >= params.rain_moisture_threshold
            water = water + jnp.where(rain, water + air_moisture, water)
            air_moisture = jnp.where(rain, 0, air_moisture)
            
            # - flow
            #   TODO: iterate if water_flow_rate * step_size is too large
            terrain, water = water_erosion_step(
                terrain, water, water_flow_rate * step_size)           
        
        step_functions.append(step)
    
    return init, *step_functions
