import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from typing import Tuple, Optional, TypeVar, Any, Union

from dirt.defaults import DEFAULT_FLOAT_TYPE
from dirt.gridworld2d.gas import step as gas_step
from dirt.gridworld2d.geology import fractal_noise
from dirt.gridworld2d.erosion import simulate_erosion_step, reset_erosion_status
from dirt.gridworld2d.water import flow_step, flow_step_twodir
from dirt.gridworld2d.naive_weather_system import weather_step
from dirt.gridworld2d.climate_pattern_day import temperature_step, light_step, get_day_status
from dirt.gridworld2d.climate_pattern_year import get_day_light_length, get_day_light_strength

from mechagogue.static_dataclass import static_dataclass

TLandscapeParams = TypeVar('TLandscapeParams', bound='LandscapeParams')
TLandscapeState = TypeVar('TLandscapeState', bound='LandscapeState')
TLandscapeAction = TypeVar('TLandscapeAction', bound='LandscapeAction')
TLandscapeObservation = TypeVar('TLanscapeObservation', bound='LandscapeObservation')


@static_dataclass
class LandscapeParams:
    world_size : Tuple[int, int] = (1024, 1024)
    step_sizes : Tuple[float, ...] = (1.)
    day_initial: int = 0.
    
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
    air_moisture_diffusion : float = 1./3.

    # Rain
    rain_moisture_up_threshold : float = SOMETHING
    rain_moisture_down_threshold: float = 0.1
    rain_amount: float = SOMETHING
    
    # air
    wind_std : float = 0.1
    wind_reversion : float = 0.001
    air_initial_temperature : float = 0.
    air_initial_smell: float = 0.

    # light
    light_initial_strength: float = SOMETHING
    night_effect: float = SOMETHING

    # temperature
    water_effect: float  = SOMETHING
    rain_effect: float = SOMETHING
    evaporation_effect: float = SOMETHING
    
    # climate
    steps_per_day : 240     # 24*10
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
    air_light: jnp.array
    air_smell: jnp.array
    rain_status: jnp.array
    day: int
    #ground_chemicals : jnp.array
    #water_chemicals : jnp.array
    #air_chemicals : jnp.array

class LandscapeAction:
    pass
    #step_size : float
    #locations : jnp.array
    #ground_chemical_update : jnp.array
    #water_chemical_update : jnp.array
    #air_chemical_update : jnp.array

class LandscapeObservation:
    pass

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
            terrain_key,
            params.world_size,
            params.terrain_octaves,
            params.terrain_lacunarity,
            params.terrain_persistence,
            params.terrain_max_octaves,
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
            water_level += params.water_initial_fill_rate # Suppose Fix
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
        air_light = jnp.zeros(params.world_size, dtype=dtype)
        air_smell = jnp.zeros(params.world_size, dtype=dtype)
        rain_status = jnp.zeros(params.world_size, dtype=dtype)
        day = params.day_initial
        
        return LandscapeState(
            terrain,
            erosion,
            water,
            wind_velocity,
            air_temperature,
            air_moisture,
            air_light,
            air_smell,
            rain_status,
            day
        )
    
    step_functions = []
    for step_size in params.step_sizes:
        def step(
            key : chex.PRNGKey,
            action : TLandscapeAction,
            state : TLandscapeState,
        ) -> Tuple[TLandscapeState, TLandscapeObservation] :
            
            terrain = state.terrain
            water = state.water
            wind_velocity = state.wind_velocity
            air_moisture = state.air_moisture
            air_smell = state.air_smell
            air_temperature = state.air_temperature

            rain_status = state.rain_status
            day_length = params.steps_per_day
            
            # apply actions

            # Day_status
            day += 1
            light_length = get_day_light_length(day)
            day_status = get_day_status(day_length, light_length, day)
            
            # move air
            # - update the wind direction
            #   TODO: iterate if step_size is too large
            key, wind_key = jrng.split(key)
            wind_velocity = step_wind_velocity(
                wind_key, step_size=step_size)
            
            # - diffuse and move the air smell
            diffusion_std = params.air_moisture_diffusion * (step_size**0.5)
            air_smell = gas_step(
                air_smell[...,None], diffusion_std, 1., wind_velocity, 1)
            
            # move water
            water = flow_step_twodir(terrain, water, params.water_flow_rate)

            # erode based on water flow
            old_terrain = terrain
            terrain, erosion = simulate_erosion_step(
                old_terrain, 
                water, 
                erosion, 
                params.water_flow_rate, 
                params.erosion_endurance, 
                params.erosion_ratio
            )
            erosion = reset_erosion_status(terrain, old_terrain, erosion)
            
            # light change based on rotation of Sun
            light_strength = get_day_light_strength(day)
            light_intensity = light_step(
                light_length, 
                terrain, water, 
                light_strength, 
                light_length, 
                day, 
                params.night_effect
            )

            # Temperature changed based on light and rain
            air_temperature = temperature_step(
                light_length, 
                day, 
                water, 
                air_temperature, 
                rain_status,
                light_intensity, 
                air_moisture,
                light_length, 
                params.night_effect, 
                params.water_effect, 
                params.rain_effect, 
                params.evaporation_effect
            )

            # Evaporate and rain based on temperature and air moisture
            water, air_evaporation, rain_status = weather_step(
                water, 
                air_temperature, 
                air_evaporation, 
                rain_status, 
                params.evaporation_rate, 
                params.rain_moisture_up_threshold, 
                params.rain_moisture_down_threshold, 
                params.rain_amount
            )
        

            # # move water
            # # - evaporate
            # #   TODO: incorporate temperature?
            # evaporation = jnp.minimum(
            #     water, params.evaporation_rate * action.step_size)
            # water = water - evaporation
            # air_moisture = air_moisture + evaporation
            
            # # - rain
            # #   TODO: incorporate temperature?
            # rain = air_moisture >= params.rain_moisture_threshold
            # water = water + jnp.where(rain, water + air_moisture, water)
            # air_moisture = jnp.where(rain, 0, air_moisture)
            
            # - flow
            #   TODO: iterate if water_flow_rate * step_size is too large

            # terrain, water = water_erosion_step(
            #     terrain, water, params.water_flow_rate * step_size)       
            
        
        step_functions.append(step)
    
    return init, *step_functions
