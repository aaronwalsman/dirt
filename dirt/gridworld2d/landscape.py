import numpy as np

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
    ICE_COLOR,
    ENERGY_TINT,
    BIOMASS_TINT,
    BIOMASS_AND_ENERGY_TINT,
)
from dirt.distribution.ou import make_ou_process
from dirt.gridworld2d.geology import fractal_noise
from dirt.gridworld2d.erosion import simulate_erosion_step, reset_erosion_status
from dirt.gridworld2d.water import flow_step, flow_step_twodir
#from dirt.gridworld2d.weather import WeatherParams, weather
# from dirt.gridworld2d.climate_pattern_day import (
#     temperature_step, light_step, get_day_status)
from dirt.gridworld2d.gas import make_gas
from dirt.gridworld2d.climate_pattern_day_cont import (
    temperature_step, light_step, get_day_status)
from dirt.gridworld2d.climate_pattern_year import (
    get_day_light_length, get_day_light_strength)
from dirt.gridworld2d.spawn import poisson_grid
from dirt.gridworld2d.grid import (
    downsample_grid,
    scale_grid,
    interpolate_grids,
    add_grids,
    take_grids,
    compare_grids,
    read_grid_locations,
    add_to_grid_locations,
    take_from_grid_locations,
)
from dirt.consumable import Consumable

@static_data
class LandscapeParams:
    '''
    space:
    Grid world units are meant to approximate a 10cm x 10cm patch of ground.
    Many systems have downsample parameters that allow them to be simulated at
    coarser resolutions.  When the values in these downsampled grids represent
    a physical quantity, they are scaled up to represent the fact each grid
    cell actually represents a larger number of more granular grid cells.
    For example if the terrain_downsample parameter is 4, then the ammount of
    water and rock in each cell will be multiplied by 16.
    
    time:
    One time step represents 6 minutes based on the defaults below.
    
    terrain:
    
    temperature:
    DIRT temperatures are approximately 1 unit = 20c
    The defaults below mean that:
     - the highest mountain peak with no sunlight will stabilize at -40c
     - the highest mountain peak with full sunlight will stabilize at 20c
     - sea level with no sunlight will stabilize at -20c
     - sea level with full sunlight will stabilize at 40c
     - sea level over water will stabilize at 30c
    
    notes on rain:
    The 'rain_per_step' parameter was approximated as 0.5mm/
    '''
    
    # space
    world_size : Tuple[int, int] = (1024, 1024)
    
    # time
    initial_time: int = 0
    steps_per_day : int = 240
    days_per_year : int = 360
    
    # terrain
    terrain_downsample : int = 4
    max_effective_altitude : float = 100.
    min_standing_water : float = 0.1
    # - rock
    include_rock : bool = True
    rock_bias : float = 0
    rock_octaves : int = 12
    rock_max_octaves : Optional[int] = None
    rock_lacunarity : float = 2.
    rock_persistence : float = 0.5
    rock_unit_scale : float = 0.005
    rock_max_height : float = 250.
    # -- erosion
    include_erosion : bool = False
    erosion_endurance : float = 0. #0.05
    erosion_ratio : float = 0. #0.01
    # - water
    include_water : bool = True
    sea_level : float = 0.
    fill_water_to_sea_level : bool = True
    initial_water_per_cell : float = 0.
    water_flow_rate : float = 0.25
    ice_flow_rate : float = 0.001
    
    # light
    include_light: bool = True
    spatial_light: bool = True
    light_initial_strength: float = 0.35
    night_effect: float = 0.15
    cloud_shade: float = 0.25
    rain_shade: float = 0.25
    
    # weather
    # - wind
    include_wind : bool = True
    wind_std : float = 3
    wind_reversion : float = 0.1
    wind_bias : Tuple[float, float] = (0., 0.)
    # - temperature
    include_temperature : bool = True
    temperature_downsample : int = 8
    initial_temperature : float = 1.
    sea_level_temperature_baseline : float = -1.
    mountain_temperature_baseline : float = -2.
    ground_heat_absorption : float = 3.
    water_heat_absorption : float = 2.
    ground_thermal_mass : float = 0.995
    water_thermal_mass : float = 0.999
    temperature_diffusion_radius : int = 1
    temperature_diffusion_strength : float = 1.
    temperature_follows_wind : bool = True
    temperature_wind_strength : float = 0.1
    # - rain
    include_rain : bool = True
    rain_downsample : int = 8
    # -- moisture
    initial_moisture : float = 0.
    evaporation_rate: float = 0.001
    min_evaporation_temp: float = 0.1
    max_evaporation_temp: float = 2.
    moisture_diffusion_radius : int = 1
    moisture_diffusion_strength : float = 1.
    # -- rain
    rain_per_step : float = 0.005
    moisture_start_raining : float = 0.5
    moisture_stop_raining : float = 0.05
    rain_altitude_scale : float = 0.25
    
    # resources
    include_resources : bool = True
    resource_downsample : int = 2
    # - energy
    initial_energy_site_density : float = (1./64.)
    initial_energy_per_site : float = 1.
    # - biomass
    initial_biomass_site_density : float = (1./64.)
    initial_biomass_per_site : float = 1.
    
    # smell
    include_smell: bool = True
    smell_downsample: int = 8
    smell_channels: int = 8
    smell_diffusion_radius: int = 1
    smell_diffusion_strength: float = 1.
    smell_dissipation: float = 0.01
    
    # audio
    include_audio: bool = True
    audio_downsample: int=32
    audio_channels: int=8
    audio_diffusion_radius: int=3
    audio_diffusion_strength: float = 1.

def make_landscape(
    params : LandscapeParams = LandscapeParams(),
    float_dtype : Any = DEFAULT_FLOAT_DTYPE,
):
    
    # check parameters
    if params.include_erosion:
        assert params.include_rock and params.include_water, (
            '"include_erosion" requires "include_rock" and "include_water"')
    
    if params.include_rain:
        assert params.include_temperature, (
            '"include_rain" requires "include_temperature"')
    
    # TODO: downsample checks
    
    # setup the wind system
    if params.include_wind:
        wind_system = make_ou_process(
            params.wind_std * jnp.sqrt(2*params.wind_reversion),
            params.wind_reversion,
            jnp.array(params.wind_bias, dtype=float_dtype),
            dtype=float_dtype,
        )
        max_wind = int(params.wind_std * 3)
    else:
        max_wind = 0
    
    # setup the temperature system
    if params.include_temperature:
        initial_temperature = (
            params.initial_temperature * (params.temperature_downsample**2))
        temperature_system = make_gas(
            params.world_size,
            downsample=params.temperature_downsample,
            initial_value=initial_temperature,
            diffusion_radius=params.temperature_diffusion_radius,
            diffusion_strength=params.temperature_diffusion_strength,
            boundary='edge',
            include_wind=params.temperature_follows_wind,
            wind_strength=params.temperature_wind_strength,
            max_wind=max_wind,
            dtype=float_dtype,
        )
   
    # setup the moisture system
    if params.include_rain:
        evaporation_temp_range = (
            params.max_evaporation_temp - params.min_evaporation_temp)
        moisture_system = make_gas(
            params.world_size,
            downsample=params.rain_downsample,
            initial_value=params.initial_moisture,
            diffusion_radius=params.moisture_diffusion_radius,
            diffusion_strength=params.moisture_diffusion_strength,
            boundary='collect',
            max_wind=max_wind,
            dtype=float_dtype,
        )
        rain_system = make_gas(
            params.world_size,
            downsample=params.rain_downsample,
            initial_value=0,
            include_diffusion=False,
            boundary='clip',
            max_wind=max_wind,
            dtype=jnp.bool,
        )
    
    # setup the smell system
    if params.include_smell:
        smell_system = make_gas(
            params.world_size,
            downsample=params.smell_downsample,
            cell_shape=(params.smell_channels,),
            initial_value=0.,
            diffusion_radius=params.smell_diffusion_radius,
            diffusion_strength=params.smell_diffusion_strength,
            dissipation=params.smell_dissipation,
            boundary='clip',
            max_wind=max_wind,
            dtype=float_dtype,
        )
    
    # setup the audio system
    if params.include_audio:
        audio_system = make_gas(
            params.world_size,
            downsample=params.audio_downsample,
            cell_shape=(params.audio_channels,),
            initial_value=0.,
            diffusion_radius=params.audio_diffusion_radius,
            diffusion_strength=params.audio_diffusion_strength,
            dissipation=1.,
            boundary='clip',
            include_wind=False,
            dtype=float_dtype,
        )
    
    # initial resource values
    mean_energy_sites = (
        params.initial_energy_site_density *
        params.world_size[0] * params.world_size[1]
    )
    mean_biomass_sites = (
        params.initial_biomass_site_density *
        params.world_size[0] * params.world_size[1]
    )
    
    @static_data
    class LandscapeState:
        
        # time
        time : int = 0
        
        # terrain
        # - rock
        if params.include_rock:
            rock : jnp.array = None
            terrain_normals: jnp.array = None
            # -- erosion
            if params.include_erosion:
                erosion : jnp.array = None
        # - water
        if params.include_water:
            water : jnp.array = None
        
        # light
        if params.include_light:
            light: jnp.array = None
        
        # weather
        # - wind
        if params.include_wind:
            wind : jnp.array = None
        # - temperature
        if params.include_temperature:
            temperature : jnp.array = None
        # - rain
        if params.include_rain:
            # -- moisture
            moisture : jnp.array = None
            # -- rain
            raining: jnp.array = None
        
        # resources
        if params.include_resources:
            # - energy
            energy : jnp.array = None
            # - biomass
            biomass : jnp.array = None
        
        # smell
        if params.include_smell:
            smell: jnp.array = None
        # audio
        if params.include_audio:
            audio: jnp.array = None
    
    @static_functions
    class Landscape:
        
        def init(
            key : chex.PRNGKey,
        ) -> LandscapeState :
            
            state = LandscapeState()
            
            # time
            state = state.replace(time=params.initial_time)
            
            # terrain
            terrain_size = (
                params.world_size[0]//params.terrain_downsample,
                params.world_size[1]//params.terrain_downsample,
            )
            
            # - rock
            if params.include_rock:
                # -- use fractal_noise to generate an initial rock grid
                key, rock_key = jrng.split(key)
                rock = fractal_noise(
                    rock_key,
                    terrain_size,
                    params.rock_octaves,
                    params.rock_lacunarity,
                    params.rock_persistence,
                    params.rock_max_octaves,
                    params.rock_unit_scale,
                    params.rock_max_height,
                    dtype=float_dtype,
                ) + params.rock_bias
                rock = rock * (params.terrain_downsample**2)
                state = state.replace(rock=rock)
                
                # -- erosion
                if params.include_erosion:
                    erosion = jnp.zeros(terrain_size, dtype=float_dtype)
                    state = state.replace(erosion=erosion)
            else:
                rock = 0.
            
            # - water
            if params.include_water:
                water = jnp.zeros(terrain_size, dtype=float_dtype)
                if params.fill_water_to_sea_level:
                    water = jnp.where(
                        rock < params.sea_level,
                        params.sea_level - rock,
                        water,
                    )
                water = water + params.initial_water_per_cell * (
                    params.terrain_downsample**2)
                state = state.replace(water=water)
            
            # light
            if params.include_light:
                if params.spatial_light:
                    light = jnp.zeros(terrain_size, dtype=float_dtype)
                    terrain_normals = jnp.zeros(
                        (*terrain_size, 3),
                        dtype=float_dtype,
                    )
                    terrain_normals = terrain_normals.at[:,:,2].set(1.)
                    # TODO: init properly
                else:
                    light = jnp.zeros((1,1), dtype=float_dtype)
                    terrain_normals = jnp.zeros((1,1,3), dtype=float_dtype)
                    terrain_normals[:,:,2] = 1.
                state = state.replace(
                    light=light, terrain_normals=terrain_normals)
            
            # weather
            # - wind
            if params.include_wind:
                key, wind_key = jrng.split(key)
                state = state.replace(wind=wind_system.init(wind_key))
            # - temperature
            if params.include_temperature:
                temperature = temperature_system.init()
                state = state.replace(temperature=temperature)
            # - rain
            if params.include_rain:
                moisture = moisture_system.init()
                mh, mw = moisture.shape[:2]
                #raining = jnp.zeros((mh, mw), dtype=jnp.bool)
                raining = rain_system.init()
                state = state.replace(moisture=moisture, raining=raining)
            
            # resources
            if params.include_resources:
                assert params.world_size[0] % params.resource_downsample == 0
                assert params.world_size[1] % params.resource_downsample == 0
                resource_size = (
                    params.world_size[0] // params.resource_downsample,
                    params.world_size[1] // params.resource_downsample,
                )
            
                # - energy
                key, energy_key = jrng.split(key)
                energy_sites = poisson_grid(
                    energy_key,
                    mean_energy_sites,
                    round(mean_energy_sites*2),
                    resource_size,
                )
                #total_energy_sites = jnp.sum(energy_sites)
                #energy_per_site = (
                #    params.initial_total_energy / total_energy_sites)
                energy = (
                    energy_sites * params.initial_energy_per_site
                ).astype(float_dtype)
                state = state.replace(energy=energy)
            
                # - biomass
                key, biomass_key = jrng.split(key)
                biomass_sites = poisson_grid(
                    biomass_key,
                    mean_biomass_sites,
                    round(mean_biomass_sites*2),
                    resource_size,
                )
                #total_biomass_sites = jnp.sum(biomass_sites)
                #biomass_per_site = (
                #    params.initial_total_biomass / total_biomass_sites)
                biomass = (
                    biomass_sites * params.initial_biomass_per_site
                ).astype(float_dtype)
                state = state.replace(biomass=biomass)
            
            # smell
            if params.include_smell:
                smell = smell_system.init()
                state = state.replace(smell=smell)
            
            # audio
            if params.include_audio:
                audio = audio_system.init()
                state = state.replace(audio=audio)
            
            return state
        
        # get/add/take water
        def get_water(state, x):
            return read_grid_locations(
                state.water, x, params.terrain_downsample)
        
        def add_water(state, x, value):
            water = add_to_grid_locations(
                state.water, x, value, params.terrain_downsample)
            return state.replace(water=water)
        
        def take_water(state, x, take=None):
            water, value = take_from_grid_locations(
                state.water, x, take, params.terrain_downsample)
            return state.replace(water=water), value
        
        # get/add/take energy
        def get_energy(state, x):
            return read_grid_locations(
                state.energy, x, params.resource_downsample)
        
        def add_energy(state, x, value):
            energy = add_to_grid_locations(
                state.energy, x, value, params.resource_downsample)
            return state.replace(energy=energy)
        
        def take_energy(state, x, take=None):
            energy, value = take_from_grid_locations(
                state.energy, x, take, params.resource_downsample)
            return state.replace(energy=energy), value
        
        # get/add/take biomass
        def get_biomass(state, x):
            return read_grid_locations(
                state.biomass, x, params.resource_downsample)
        
        def add_biomass(state, x, value):
            biomass = add_to_grid_locations(
                state.biomass, x, value, params.resource_downsample)
            return state.replace(biomass=biomass)
        
        def take_biomass(state, x, take=None):
            biomass, value = take_from_grid_locations(
                state.biomass, x, take, params.resource_downsample)
            return state.replace(biomass=biomass), value
        
        def step(
            key : chex.PRNGKey,
            state : LandscapeState,
        ) -> LandscapeState :
            
            # time
            t = state.time + 1
            state = state.replace(time=t)
            
            light_length = get_day_light_length(t)
            day_status = get_day_status(params.steps_per_day, light_length, t)
            
            # water
            if params.include_water:
                if params.include_temperature:
                    flow_rate = jnp.where(
                        state.temperature < 0.,
                        params.ice_flow_rate,
                        params.water_flow_rate,
                    )
                else:
                    flow_rate = jnp.array(
                        params.water_flow_rate, dtype=float_dtype).reshape(1,1)
                water = flow_step(state.rock, state.water, flow_rate)
                state = state.replace(water=water)
                standing_water = (
                    state.water >
                    params.min_standing_water * params.terrain_downsample**2
                )
            
            # rock and erosion
            if params.include_rock and params.include_erosion:
                old_rock = state.rock
                rock, erosion = simulate_erosion_step(
                    old_rock,
                    state.water,
                    state.erosion,
                    params.water_flow_rate,
                    params.erosion_endurance,
                    params.erosion_ratio
                )
                erosion = reset_erosion_status(rock, old_rock, erosion)
                state = state.replace(rock=rock, erosion=erosion)
            
            # compute altitude
            altitude = jnp.zeros((), dtype=float_dtype)
            normalized_altitude = jnp.zeros((), dtype=float_dtype)
            if params.include_rock:
                altitude += state.rock
            if params.include_water:
                altitude += state.water
            normalized_altitude = jnp.clip(
                altitude/params.max_effective_altitude, 0., 1.)
            
            # wind
            if params.include_wind:
                key, wind_key = jrng.split(key)
                wind = wind_system.step(wind_key, state.wind)
                state = state.replace(wind=wind)
            
            # rain
            if params.include_rain:
                # compute evaporation (water -> moisture)
                # - use the temperature to figure out how much is evaporating
                evaporation = (
                    (state.temperature - params.min_evaporation_temp) /
                    evaporation_temp_range
                )
                evaporation = jnp.clip(evaporation, min=0., max=1.)
                # - turn off evaporation wherever it's raining
                evaporation = scale_grid(
                    evaporation, jnp.logical_not(state.raining))
                evaporation *= (
                    params.evaporation_rate * (params.rain_downsample**2))
                # - remove the evaporation from the water
                water, evaporated = take_grids(state.water, evaporation)
                # - add the evaporation to the moisture
                moisture = add_grids(state.moisture, evaporated)
                
                # compute rain (moisture -> water)
                rain_ammount = jnp.where(
                    state.raining, params.rain_per_step, 0.)
                moisture, rained = take_grids(moisture, rain_ammount)
                water = add_grids(water, rained)
                
                # compute raining (where to rain next)
                altitude_rain_scale = (
                    1. - normalized_altitude + params.rain_altitude_scale)
                # - where should it start raining
                moisture_start_raining = (
                    params.moisture_start_raining * altitude_rain_scale)
                start_raining = compare_grids(
                    moisture, moisture_start_raining, mode='>')
                # - where should it stop raining
                moisture_stop_raining = (
                    params.moisture_stop_raining * altitude_rain_scale)
                stop_raining = compare_grids(
                    moisture, moisture_stop_raining, mode='<=')
                # - update raining
                raining = (state.raining | start_raining) & ~stop_raining
                # - smooth out the raining pattern
                kernel = jnp.array([
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ], dtype=jnp.int8).reshape((3,3,1,1))
                raining_neighbors = jax.lax.conv_general_dilated(
                    raining.astype(jnp.int8)[None,...,None],
                    kernel,
                    window_strides=(1,1),
                    padding='SAME',
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                )[0,...,0]
                raining = (
                    (raining & raining_neighbors > 3) | raining_neighbors > 6)
                raining = raining.astype(jnp.bool)
                
                # apply rain gas dynamics
                key, rain_key = jrng.split(key)
                raining = rain_system.step(rain_key, raining, wind=state.wind)
                
                # apply moisture gas dynamics
                key, moisture_key = jrng.split(key)
                moisture = moisture_system.step(
                    moisture_key, moisture, wind=state.wind)
                
                # update state
                state = state.replace(
                    water=water, moisture=moisture, raining=raining)
            
            # compute light change based on rotation of the sun
            if params.include_light:
                light_strength = get_day_light_strength(t)
                light = light_step(
                    params.steps_per_day,
                    altitude, 
                    light_strength,
                    light_length,
                    t,
                    params.night_effect
                )
                light = light * (params.terrain_downsample**2)
                state = state.replace(light=light)
            
            # reduce light based on shade from clouds and rain
            if params.include_rain:
                clouds = scale_grid(moisture, 1./moisture_start_raining)
                shade = 1. - (
                    clouds*params.cloud_shade + raining*params.rain_shade)
                light = scale_grid(light, shade)
            
            
            if params.include_temperature:
                # apply temperature gas dynamics
                key, temperature_key = jrng.split(key)
                temperature = temperature_system.step(
                    temperature_key, state.temperature, wind=wind)
                
                # update the temperature based on the incoming light
                # - compute the temperature blend
                temperature_alpha = jnp.where(
                    standing_water,
                    params.water_thermal_mass,
                    params.ground_thermal_mass,
                )
                # - compute the target temperature
                a = normalized_altitude
                temperature_baseline = (
                    a * params.mountain_temperature_baseline +
                    (1. - a) * params.sea_level_temperature_baseline
                )
                # - compute the heat absorption
                heat_absorption = jnp.where(
                    standing_water,
                    params.water_heat_absorption,
                    params.ground_heat_absorption,
                )
                # - correct for light not being full strength all day
                c = jnp.array((4./jnp.pi), dtype=float_dtype)
                target_temperature = (
                    temperature_baseline + c * light * heat_absorption)
                
                # - interpolate between the current and target temperature
                temperature = interpolate_grids(
                    temperature, target_temperature, temperature_alpha)
                
                # update state
                state = state.replace(temperature=temperature)
            
            return state
        
        def render(state, downsample):
            h = params.world_size[0]//downsample
            w = params.world_size[1]//downsample
            
            # make everything rock colored
            rgb = jnp.full((h,w,3), ROCK_COLOR, dtype=float_dtype)
            
            # overlay water and ice
            th, tw = state.temperature.shape
            #temperature_scale = float(h/th)
            water_color = jnp.where(
                state.temperature[..., None] <= 0.,
                ICE_COLOR,
                WATER_COLOR,
            )# * (temperature_scale**2)
            
            standing_water = (
                downsample_grid(state.water[..., None], th, tw) >= (
                params.min_standing_water * (params.terrain_downsample**2))
            )
            rgb = interpolate_grids(
                rgb, water_color, ~standing_water, preserve_mass=False)
            # apply the energy and biomass tint
            # - compute resource_downsample here instead of using params
            #   because when using this for reports, it may be downsampled
            #   further
            rh = state.energy.shape[0]
            resource_downsample = params.world_size[0] // rh
            
            max_resource = resource_downsample**2
            clipped_energy = jnp.clip(state.energy, min=0, max=max_resource)
            clipped_biomass = jnp.clip(state.biomass, min=0, max=max_resource)
            biomass_and_energy = jnp.minimum(
                clipped_energy, clipped_biomass)
            just_energy = clipped_energy - biomass_and_energy
            just_biomass = clipped_biomass - biomass_and_energy
            
            rgb = add_grids(
                rgb,
                biomass_and_energy[..., None] * BIOMASS_AND_ENERGY_TINT,
                #preserve_mass=False,
            )
            rgb = add_grids(
                rgb,
                just_energy[..., None] * ENERGY_TINT,
                #preserve_mass=False,
            )
            rgb = add_grids(
                rgb,
                just_biomass[..., None] * BIOMASS_TINT,
                #preserve_mass=False,
            )
            
            #if object_x is not None:
            #    rgb = rgb.at[object_x[..., 0], object_x[..., 1]].set(
            #        object_color)
            
            #light_scale = (params.world_size[0]/h)/(params.terrain_downsample)
            #light_scale = light_scale**2
            light_scale = 1./(params.terrain_downsample**2)
            rgb = scale_grid(rgb, state.light[..., None]*light_scale)
            rgb = np.array(jnp.clip(rgb, min=0., max=1.) * 255).astype(np.uint8)
            
            return rgb
    
    return Landscape
