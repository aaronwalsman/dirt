import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from typing import Tuple, Optional, TypeVar, Any, Union

from mechagogue.static import static_data, static_functions

from dirt.constants import (
    DEFAULT_FLOAT_DTYPE,
    ROCK_COLOR,
    WATER_COLOR,
    ENERGY_TINT,
    BIOMASS_TINT,
)
#from dirt.gridworld2d.gas import step as gas_step
#from dirt.distribution.ou import ou_process
from dirt.gridworld2d.geology import fractal_noise
from dirt.gridworld2d.erosion import simulate_erosion_step, reset_erosion_status
from dirt.gridworld2d.water import flow_step, flow_step_twodir
from dirt.gridworld2d.weather import WeatherParams, make_weather
# from dirt.gridworld2d.climate_pattern_day import (
#     temperature_step, light_step, get_day_status)
from dirt.gridworld2d.climate_pattern_day_year import (
    temperature_step, light_step, get_day_status)
from dirt.gridworld2d.climate_pattern_year import (
    get_day_light_length, get_day_light_strength)
from dirt.gridworld2d.spawn import poisson_grid
from dirt.consumable import Consumable

@static_data
class LandscapeParams:
    world_size : Tuple[int, int] = (1024, 1024)
    #step_sizes : Tuple[float, ...] = (1.,)
    initial_time: float = 0.
    
    # terrain
    terrain_bias : float = 0
    terrain_octaves : int = 12
    terrain_max_octaves : int = None
    terrain_lacunarity : float = 2.
    terrain_persistence : float = 0.5
    terrain_unit_scale : float = 0.005
    terrain_max_height : float = 50.
    max_effective_altitude : float = 100.
    
    # water
    include_water_flow : bool = True
    
    # - different ways to fill the water
    sea_level : float = 0.
    initial_water_per_cell : float = 0.
    
    water_flow_rate : float = 1
    ice_flow_rate : float = 0.001
    air_moisture_diffusion : float = 0.333
    min_effective_water : float = 0.05
    
    # smell
    smell_downsample: int = 1
    smell_channels: int = 8
    
    # audio
    audio_downsample: int = 32
    audio_channels: int = 8
    
    # light
    include_light: bool = True
    light_initial_strength: float = 0.35
    night_effect: float = 0.15
    cloud_shade: float = 0.25
    rain_shade: float = 0.25
    
    # temperature
    water_effect: float  = 0.25
    rain_effect: float = 0.05
    evaporation_effect: float = 0.05
    
    # climate
    steps_per_day : int = 240     # 24*10
    days_per_year : int = 360
    max_season_angle: float = 0.41
    #evaporation_rate : float = 0.01
    
    # weather
    weather : WeatherParams = WeatherParams()
    
    # erosion
    erosion_endurance : float = 0. #0.05
    erosion_ratio : float = 0. #0.01
    
    # energy
    mean_energy_sites : float = 256.**2
    initial_total_energy : float = 256.**2
    
    # biomass
    mean_biomass_sites : float = 256.**2
    initial_total_biomass : float = 256.**2

@static_data
class LandscapeState:
    terrain : jnp.array
    erosion : jnp.array
    water : jnp.array
    temperature : jnp.array
    moisture : jnp.array
    rain: jnp.array
    wind : jnp.array
    light: jnp.array
    smell: jnp.array
    audio: jnp.array
    time: float
    
    energy : jnp.array
    biomass : jnp.array

def make_landscape(
    params : LandscapeParams = LandscapeParams(),
    float_dtype : Any = DEFAULT_FLOAT_DTYPE,
):
    
    #step_weathers = []
    #for step_size in params.step_sizes:
    #weather_params = params.weather.replace(step_size=step_size)
    weather = make_weather(params.weather, float_dtype=float_dtype)
    #step_weathers.append(step_weather)
    
    @static_functions
    class Landscape:
        def init(
            key : chex.PRNGKey,
        ) -> LandscapeState :
            
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
                dtype=float_dtype,
            ) + params.terrain_bias
            
            # erosion
            # - initialize erosion to be zero everywhere
            erosion = jnp.zeros(params.world_size, dtype=float_dtype)
            
            # water
            #water = jnp.max(params.sea_level - terrain, 0.)
            water = jnp.where(
                terrain < params.sea_level, params.sea_level - terrain, 0.)
            water = water + params.initial_water_per_cell
            '''
            # - start with zero water everywhere
            water = jnp.zeros(params.world_size, dtype=float_dtype)
            
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
                water = water.astype(float_dtype)
                return (water, water_level)
            
            water, water_level = jax.lax.while_loop(
                water_level_cond, water_level_body, (water, water_level))
            
            # - scale the water everywhere in order to counteract any overshoot
            current_water = jnp.sum(water)
            water = water * (total_water / current_water)
            
            # - offset the terrain so that the water level (sea level) is zero
            terrain = terrain - water_level
            '''
            # light
            light = jnp.zeros(params.world_size, dtype=float_dtype)
            
            # smell
            smell_size = (
                params.world_size[0]//params.smell_downsample,
                params.world_size[1]//params.smell_downsample,
            )
            smell = jnp.zeros(
                (*smell_size, params.smell_channels), dtype=float_dtype)
            
            # audio
            audio_size = (
                params.world_size[0]//params.audio_downsample,
                params.world_size[1]//params.audio_downsample,
            )
            audio = jnp.zeros(
                (*audio_size, params.audio_channels), dtype=float_dtype)
            
            # weather
            key, weather_key = jrng.split(key)
            temperature, moisture, rain, wind = init_weather(weather_key)
            
            # time
            t = params.initial_time
            
            # energy
            key, energy_key = jrng.split(key)
            energy_sites = poisson_grid(
                energy_key,
                params.mean_energy_sites,
                round(params.mean_energy_sites*2),
                params.world_size,
            )
            total_energy_sites = jnp.sum(energy_sites)
            energy_per_site = params.initial_total_energy / total_energy_sites
            energy = (energy_sites * energy_per_site).astype(float_dtype)
            
            # biomass
            key, biomass_key = jrng.split(key)
            biomass_sites = poisson_grid(
                biomass_key,
                params.mean_biomass_sites,
                round(params.mean_biomass_sites*2),
                params.world_size,
            )
            total_biomass_sites = jnp.sum(biomass_sites)
            biomass_per_site = (
                params.initial_total_biomass / total_biomass_sites)
            biomass = (biomass_sites * biomass_per_site).astype(float_dtype)
            
            return LandscapeState(
                terrain,
                erosion,
                water,
                temperature,
                moisture,
                rain,
                wind,
                light,
                smell,
                audio,
                t,
                energy,
                biomass,
            )
        
        def get_consumable(state, x):
            x0, x1 = x[...,0], x[...,1]
            water = state.water[x0, x1]
            energy = state.energy[x0, x1]
            biomass = state.biomass[x0, x1]
            return Consumable(water, energy, biomass)
        
        def set_consumable(state, x, consumable):
            x0, x1 = x[...,0], x[...,1]
            water = state.water.at[x0, x1].set(consumable.water)
            energy = state.energy.at[x0, x1].set(consumable.energy)
            biomass = state.biomass.at[x0, x1].set(consumable.biomass)
            return state.replace(water=water, energy=energy, biomass=biomass)
        
        def add_consumable(state, x, consumable):
            x0, x1 = x[...,0], x[...,1]
            water = state.water.at[x0, x1].add(consumable.water)
            energy = state.energy.at[x0, x1].add(consumable.energy)
            biomass = state.biomass.at[x0, x1].add(consumable.biomass)
            return state.replace(water=water, energy=energy, biomass=biomass)
        
        #step_functions = []
        #for i, step_size in enumerate(params.step_sizes):
        def step(
            key : chex.PRNGKey,
            state : LandscapeState,
        ) -> LandscapeState :
            
            terrain = state.terrain
            water = state.water
            erosion = state.erosion
            wind = state.wind
            moisture = state.moisture
            rain = state.rain
            smell = state.smell
            audio = state.audio
            temperature = state.temperature
            t = state.time

            day_length = params.steps_per_day
            days_per_year = params.days_per_year
            max_season_angle = params.max_season_angle

            # Calulate the season angle based on the time
            total_steps_per_year = day_length * days_per_year
            season_angle = max_season_angle * jnp.sin(2 * jnp.pi * t / total_steps_per_year)

            
            # Day_status
            t += step_size
            light_length = get_day_light_length(t)
            day_status = get_day_status(day_length, light_length, t)
            
            # TODO: Concretization problem... need to configure the
            # kernel and not have it dynamically shaped
            #smell = gas_step(
            #    smell, diffusion_std, 1., wind, 1)
            key, smell_key = jrng.split(key)
            #smell = smell_step(smell_key, smell, wind=
            
            # move water
            if params.include_water_flow:
                flow_rate = jnp.where(
                    temperature < 0.,
                    params.ice_flow_rate * step_size,
                    params.water_flow_rate * step_size,
                )
                water = flow_step(terrain, water, flow_rate)

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
            
            altitude = terrain + water
            normalized_altitude = jnp.clip(
                altitude/params.max_effective_altitude, 0., 1.)
            
            # light change based on rotation of Sun
            if params.include_light:
                light_strength = get_day_light_strength(t)
                light = light_step(
                    day_length,
                    altitude, 
                    light_strength,
                    light_length,
                    t,
                    season_angle,
                    params.night_effect
                )
            else:
                light = jnp.ones((), dtype=float_dtype)
            
            # reduce light based on shade from clouds and rain
            clouds = moisture / params.weather.moisture_start_raining
            shade = 1. - (clouds*params.cloud_shade + rain*params.rain_shade)
            light = light * shade
            
            # temperature changed based on light and rain
            #temperature = temperature_step(
            #    day_length, 
            #    t, 
            #    water, 
            #    temperature, 
            #    rain,
            #    light, 
            #    moisture,
            #    light_length, 
            #    params.night_effect, 
            #    params.water_effect, 
            #    params.rain_effect, 
            #    params.evaporation_effect
            #)

            # evaporate and rain based on temperature and air moisture
            #water, moisture, rain = weather_step(
            #    water,
            #    temperature,
            #    moisture,
            #    rain,
            #    params.evaporation_rate,
            #    params.rain_moisture_up_threshold,
            #    params.rain_moisture_down_threshold,
            #    params.rain_amount,
            #    wind,
            #)
            key, weather_key = jrng.split(key)
            (
                water,
                temperature,
                moisture,
                rain,
                wind,
                discrete_wind,
            ) = step_weathers[i](
                weather_key,
                water,
                temperature,
                moisture,
                rain,
                wind,
                normalized_altitude,
                light,
            )
            
            next_state = state.replace(
                terrain=terrain,
                erosion=erosion,
                water=water,
                temperature=temperature,
                moisture=moisture,
                rain=rain,
                wind=wind,
                light=light,
                #smell=smell,
                time=t,
            )
            
            return next_state
            
            #step_functions.append(step)
    
    return Landscape

if __name__ == "__main__":
    init, step_fn = landscape()
    key = jax.random.PRNGKey(1234)
    state = init(key)
    for i in range(500):
        key, subkey = jax.random.split(key)
        state = step_fn(subkey, state)
        if i % 20 == 0:
            # inspect
            print(f"\n--- Day {state.time} ---")
            print("Wind velocity:", state.wind)
            print("Air temperature (mean):", jnp.mean(state.temperature))
            print("Water (sum):", jnp.sum(state.water + state.moisture))
            print("Rain status (mean):", jnp.mean(state.rain))
            print("Erosion (mean):", jnp.mean(state.erosion))
