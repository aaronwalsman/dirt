import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from typing import Tuple, Optional, TypeVar, Any, Union

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
from dirt.gridworld2d.weather import WeatherParams, weather
# from dirt.gridworld2d.climate_pattern_day import (
#     temperature_step, light_step, get_day_status)
from dirt.gridworld2d.climate_pattern_day_cont import (
    temperature_step, light_step, get_day_status)
from dirt.gridworld2d.climate_pattern_year import (
    get_day_light_length, get_day_light_strength)
from dirt.gridworld2d.spawn import poisson_grid
from dirt.consumable import Consumable

from mechagogue.static_dataclass import static_dataclass

@static_dataclass
class LandscapeParams:
    world_size : Tuple[int, int] = (1024, 1024)
    step_sizes : Tuple[float, ...] = (1.,)
    day_initial: int = 0.
    
    # terrain
    terrain_octaves : int = 12
    terrain_max_octaves : Optional[int] = None
    terrain_lacunarity : float = 2.
    terrain_persistence : float = 0.5
    terrain_unit_scale : float = 0.005
    terrain_max_height : float = 50.
    
    # water
    water_per_cell : float = 2.
    water_initial_fill_rate : float = 0.01
    water_flow_rate : float = 0.1
    air_moisture_diffusion : float = 1./3.

    # rain
    # moved to weather
    #rain_moisture_up_threshold : float = 0.8
    #rain_moisture_down_threshold: float = 0.4
    #rain_amount: float = 0.32
    
    # air
    # moved to weather
    #wind_std : float = 0.1
    #wind_reversion : float = 0.001
    #wind_bias : Tuple[float, float] | jnp.ndarray = (0.,0.)
    #air_initial_temperature : float = 0.
    air_initial_smell: float = 0.

    # light
    light_initial_strength: float = 0.35
    night_effect: float = 0.15
    
    # temperature
    water_effect: float  = 0.25
    rain_effect: float = 0.05
    evaporation_effect: float = 0.05
    
    # climate
    steps_per_day : int = 240     # 24*10
    days_per_year : int = 360
    #evaporation_rate : float = 0.01
    
    # weather
    weather : WeatherParams = WeatherParams()
    
    # erosion
    erosion_endurance : float = 0.05
    erosion_ratio : float = 0.01
    
    # energy
    mean_energy_sites : float = 256.**2
    initial_total_energy : float = 256.**2
    
    # biomass
    mean_biomass_sites : float = 256.**2
    initial_total_biomass : float = 256.**2

@static_dataclass
class LandscapeState:
    terrain : jnp.array
    erosion : jnp.array
    water : jnp.array
    temperature : jnp.array
    moisture : jnp.array
    rain: jnp.array
    wind : jnp.array
    air_light: jnp.array
    air_smell: jnp.array
    day: int
    
    energy : jnp.array
    biomass : jnp.array

def landscape(
    params : LandscapeParams = LandscapeParams(),
    float_dtype : Any = DEFAULT_FLOAT_DTYPE,
):
    
    # moved to weather
    #init_wind, step_wind = ou_process(
    #    params.wind_std,
    #    params.wind_reversion,
    #    jnp.zeros((2,), dtype=float_dtype),
    #    dtype=float_dtype,
    #)
    
    step_weathers = []
    for step_size in params.step_sizes:
        init_weather, step_weather = weather(
            params.weather, step_size=step_size, float_dtype=float_dtype)
        step_weathers.append(step_weather)
    
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
        )
        
        # erosion
        # - initialize erosion to be zero everywhere
        erosion = jnp.zeros(params.world_size, dtype=float_dtype)
        
        # water
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
        
        # air
        #temperature = jnp.full(
        #    params.world_size,
        #    params.air_initial_temperature,
        #    dtype=float_dtype,
        #)
        air_light = jnp.zeros(params.world_size, dtype=float_dtype)
        air_smell = jnp.zeros(params.world_size, dtype=float_dtype)
        air_smell = air_smell[...,None]
        
        # weather
        key, weather_key = jrng.split(key)
        temperature, moisture, rain, wind = init_weather(weather_key)
        
        day = params.day_initial
        
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
        
        key, biomass_key = jrng.split(key)
        biomass_sites = poisson_grid(
            biomass_key,
            params.mean_biomass_sites,
            round(params.mean_biomass_sites*2),
            params.world_size,
        )
        total_biomass_sites = jnp.sum(biomass_sites)
        biomass_per_site = params.initial_total_biomass / total_biomass_sites
        biomass = (biomass_sites * biomass_per_site).astype(float_dtype)
        
        return LandscapeState(
            terrain,
            erosion,
            water,
            temperature,
            moisture,
            rain,
            wind,
            air_light,
            air_smell,
            day,
            energy,
            biomass,
        )
    
    def get_consumable(state, locations):
        y = locations[...,0]
        x = locations[...,1]
        water = state.water[y, x]
        energy = state.energy[y, x]
        biomass = state.biomass[y, x]
        return Consumable(water, energy, biomass)
    
    def set_consumable(state, locations, consumable):
        y, x = locations[...,0], locations[...,1]
        water = state.water.at[y, x].set(consumable.water)
        energy = state.energy.at[y, x].set(consumable.energy)
        biomass = state.biomass.at[y, x].set(consumable.biomass)
        return state.replace(water=water, energy=energy, biomass=biomass)
    
    def add_consmable(state, locations, consumable):
        y, x = locations[...,0], locations[...,1]
        water = state.water.at[y, x].add(consumable.water)
        energy = state.energy.at[y, x].add(consumable.energy)
        biomass = state.biomass.at[y, x].add(consumable.biomass)
        return state.replace(water=water, energy=energy, biomass=biomass)
    
    step_functions = []
    for i, step_size in enumerate(params.step_sizes):
        def step(
            key : chex.PRNGKey,
            state : LandscapeState,
        ) -> LandscapeState :
            
            terrain = state.terrain
            water = state.water
            erosion = state.erosion
            wind = state.wind
            moisture = state.moisture
            air_smell = state.air_smell
            temperature = state.temperature
            day = state.day

            rain = state.rain
            day_length = params.steps_per_day
            
            # Day_status
            day += 1
            light_length = get_day_light_length(day)
            day_status = get_day_status(day_length, light_length, day)
            
            # move air
            # - update the wind direction
            #   TODO: iterate if step_size is too large
            # moved to weather
            #key, wind_key = jrng.split(key)
            #wind = step_wind(
            #    wind_key, wind, step_size=step_size)
            
            # - diffuse and move the air smell
            diffusion_std = params.air_moisture_diffusion * (step_size**0.5)
            
            # TODO: Concretization problem... need to configure the
            # kernel and not have it dynamically shaped
            #air_smell = gas_step(
            #    air_smell, diffusion_std, 1., wind, 1)
            
            # move water
            water = flow_step(terrain, water, params.water_flow_rate)

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
            light = light_step(
                day_length,
                terrain, water, 
                light_strength,
                light_length,
                day,
                params.night_effect
            )
            
            # temperature changed based on light and rain
            #temperature = temperature_step(
            #    day_length, 
            #    day, 
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
            water, temperature, moisture, rain, wind = step_weathers[i](
                weather_key,
                water,
                temperature,
                moisture,
                rain,
                wind,
                terrain,
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
                air_light=light,
                air_smell=air_smell,
                day=day,
            )
            
            return next_state
        
        step_functions.append(step)
    
    return init, get_consumable, set_consumable, add_consumable, *step_functions


if __name__ == "__main__":
    init, step_fn = landscape()
    key = jax.random.PRNGKey(1234)
    state = init(key)
    for i in range(500):
        key, subkey = jax.random.split(key)
        state = step_fn(subkey, state)
        if i % 20 == 0:
            # inspect
            print(f"\n--- Day {state.day} ---")
            print("Wind velocity:", state.wind)
            print("Air temperature (mean):", jnp.mean(state.temperature))
            print("Water (sum):", jnp.sum(state.water + state.moisture))
            print("Rain status (mean):", jnp.mean(state.rain))
            print("Erosion (mean):", jnp.mean(state.erosion))
