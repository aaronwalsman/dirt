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
    light_step, get_day_status)
from dirt.gridworld2d.climate_pattern_year import (
    get_day_light_length, get_day_light_strength)
from dirt.gridworld2d.spawn import unique_x, poisson_grid
from dirt.gridworld2d.grid import (
    downsample_grid_shape,
    zero_grid,
    downsample_grid,
    subsample_grid,
    upsample_grid,
    set_grid_shape,
    grid_sum_to_mean,
    grid_mean_to_sum,
    scale_grid,
    interpolate_grids,
    add_grids,
    take_grids,
    compare_grids,
    read_grid_locations,
    write_grid_locations,
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
    min_standing_water : float = 0.01
    # - rock
    include_rock : bool = True
    rock_bias : float = 0
    rock_octaves : int = 12
    rock_max_octaves : int = 20
    rock_lacunarity : float = 2.
    rock_persistence : float = 0.5
    rock_unit_scale : float = 0.005
    rock_max_height : float = 50.
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
    include_water_sources_and_sinks : bool = True
    water_sink_density : float = 10./(128**2)
    water_sink_flow : float = 0.1
    water_source_density : float = 1./(128**2)
    water_source_flow : float = 1.
    
    # light
    include_light: bool = True
    light_downsample : int = 8
    light_initial_strength: float = 0.35
    night_effect: float = 0.15
    cloud_shade: float = 0.25
    rain_shade: float = 0.25
    
    # weather
    # - wind
    include_wind : bool = True
    wind_std : float = 5
    wind_reversion : float = 0.05
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
    initial_energy_site_density : float = (1./16.)
    initial_energy_per_site : float = 1.
    # - biomass
    initial_biomass_site_density : float = (1./16.)
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

    def validate(params):
        
        def validate_downsample(downsample, world_size):
            downsample = min(downsample, world_size[0], world_size[1])
            assert (
                world_size[0] % downsample == 0 and
                world_size[1] % downsample == 0
            )
            return downsample
        
        if params.include_erosion:
            assert params.include_rock and params.include_water, (
                '"include_erosion" requires "include_rock" and "include_water"')
        
        if params.include_rain:
            assert params.include_temperature, (
                '"include_rain" requires "include_temperature"')
        
        if params.include_light:
            light_downsample = max(
                params.terrain_downsample, params.light_downsample)
        
        if params.include_temperature:
            assert params.include_light, (
                '"include_temperature" requires "include_light"')
        
        return params.replace(
            terrain_downsample=validate_downsample(
                params.terrain_downsample, params.world_size),
            light_downsample=validate_downsample(
                light_downsample, params.world_size),
            temperature_downsample=validate_downsample(
                params.temperature_downsample, params.world_size),
            rain_downsample=validate_downsample(
                params.rain_downsample, params.world_size),
            resource_downsample=validate_downsample(
                params.resource_downsample, params.world_size),
            smell_downsample=validate_downsample(
                params.smell_downsample, params.world_size),
            audio_downsample=validate_downsample(
                params.audio_downsample, params.world_size),
        )

def make_landscape(
    params : LandscapeParams = LandscapeParams(),
    float_dtype : Any = DEFAULT_FLOAT_DTYPE,
):
    
    # validate parameters
    params = params.validate()
    
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
   
    # setup the rain system
    if params.include_rain:
        evaporation_temp_range = (
            params.max_evaporation_temp - params.min_evaporation_temp)
        # - moisture
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
        # - rain
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
    
    def compute_normalized_altitude(state):
        altitude = jnp.zeros((), dtype=float_dtype)
        if params.include_rock:
            altitude += state.rock
        if params.include_water:
            altitude += state.water
        altitude = grid_sum_to_mean(altitude, params.terrain_downsample)
        normalized_altitude = (
            (altitude - params.sea_level) / params.max_effective_altitude)
        normalized_altitude = jnp.clip(normalized_altitude, 0., 1.)
        
        return normalized_altitude
    
    @static_data
    class LandscapeState:
        
        # time
        time : int = 0
        
        # terrain
        # - rock
        if params.include_rock:
            rock : jnp.ndarray = None
            rock_normals: jnp.ndarray = None
            # -- erosion
            if params.include_erosion:
                erosion : jnp.ndarray = None
        # - water
        if params.include_water:
            water : jnp.ndarray = None
            water_source_locations : jnp.ndarray = None
            water_sink_locations : jnp.ndarray = None
        
        # light
        if params.include_light:
            light: jnp.ndarray = None
        
        # weather
        # - wind
        wind : jnp.ndarray = None
        # - temperature
        temperature : jnp.ndarray = None
        # - rain
        # -- moisture
        moisture : jnp.ndarray = None
        # -- rain
        raining : jnp.ndarray = None
        
        # resources
        # - energy
        energy : jnp.ndarray = None
        # - biomass
        biomass : jnp.ndarray = None
        
        # smell
        smell: jnp.ndarray = None
        # audio
        audio: jnp.ndarray = None
    
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
                    params.rock_unit_scale * params.terrain_downsample,
                    params.rock_max_height,
                    dtype=float_dtype,
                ) + params.rock_bias
                rock = grid_mean_to_sum(rock, params.terrain_downsample)
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
                    downsampled_sea_level = (
                        params.sea_level * params.terrain_downsample**2)
                    water = jnp.where(
                        rock < downsampled_sea_level,
                        params.sea_level - rock,
                        water,
                    )
                water = (
                    water +
                    params.initial_water_per_cell * params.terrain_downsample**2
                )
                state = state.replace(water=water)
                
                # -- water sink and sources
                if params.include_water_sources_and_sinks:
                    assert params.water_source_density > 0
                    num_water_sources = jnp.ceil(
                        params.water_source_density *
                        params.world_size[0] * params.world_size[1]
                    )
                    assert params.water_sink_density > 0
                    num_water_sinks = jnp.ceil(
                        params.water_sink_density *
                        params.world_size[0] * params.world_size[1]
                    )
                    key, water_source_key, water_sink_key = jrng.split(key, 3)
                    water_source_locations = unique_x(
                        water_source_key, num_water_sources, params.world_size)
                    water_sink_locations = unique_x(
                        water_sink_key, num_water_sinks, params.world_size)
                    state = state.replace(
                        water_source_locations=water_source_locations,
                        water_sink_locations=water_sink_locations,
                    )
            
            # light
            if params.include_light:
                light_shape = downsample_grid_shape(
                    *params.world_size, params.light_downsample)
                if light_shape[0] == 1 or light_shape[1] == 1:
                    rock_normals = jnp.full(
                        (1,1,3), jnp.array([0,0,1], dtype=float_dtype))
                else:
                    resampled_rock = set_grid_shape(
                        state.rock, *light_shape)
                    resampled_rock = grid_sum_to_mean(
                        resampled_rock, params.light_downsample)
                    dx, dy = jnp.gradient(resampled_rock)
                    dx /= params.light_downsample
                    dy /= params.light_downsample
                    rock_normals = jnp.stack(
                        (-dx, -dy, jnp.ones_like(resampled_rock)), axis=-1)
                    magnitude = jnp.linalg.norm(
                        rock_normals, axis=-1, keepdims=True)
                    magnitude = jnp.clip(magnitude, min=1e-8)
                    rock_normals = rock_normals / magnitude
                
                # TODO: use the light calculation for the first step instead
                # of zeros
                light = zero_grid(
                    *params.world_size,
                    params.light_downsample,
                    dtype=float_dtype,
                )
                
                state = state.replace(
                    light=light, rock_normals=rock_normals)
            
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
                    params.world_size,
                )
                energy = (
                    energy_sites * params.initial_energy_per_site
                ).astype(float_dtype)
                energy = downsample_grid(energy, *resource_size)
                state = state.replace(energy=energy)
            
                # - biomass
                key, biomass_key = jrng.split(key)
                biomass_sites = poisson_grid(
                    biomass_key,
                    mean_biomass_sites,
                    round(mean_biomass_sites*2),
                    params.world_size,
                )
                biomass = (
                    biomass_sites * params.initial_biomass_per_site
                ).astype(float_dtype)
                biomass = downsample_grid(biomass, *resource_size)
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
                # - make water flow downhill
                if params.include_temperature:
                    flow_rate = jnp.where(
                        state.temperature > 0.,
                        params.water_flow_rate,
                        params.ice_flow_rate,
                    )
                else:
                    flow_rate = jnp.full(
                        (1,1), params.water_flow_rate, dtype=float_dtype)
                flow_rate /= params.terrain_downsample
                water = flow_step(state.rock, state.water, flow_rate)
                
                # - sources and sinks
                #   The total ammount of water transferred is equal to the
                #   minimum of the number of unfrozen sinks times the
                #   water_sink_flow parameter and the number of unfrozen
                #   sources times the water_source_flow parameter.  This value
                #   is divided equally between the number of unfrozen sources
                #   and the number of unfrozen sinks.
                if params.include_water_sources_and_sinks:
                    num_water_sinks = state.water_sink_locations.shape[0]
                    num_water_sources = state.water_source_locations.shape[0]
                    if params.include_temperature:
                        # -- determine how many sinks and sources are frozen
                        sinks_unfrozen = read_grid_locations(
                            state.temperature,
                            state.water_sink_locations,
                            params.temperature_downsample,
                        ) > 0.
                        sources_unfrozen = read_grid_locations(
                            state.temperature,
                            state.water_source_locations,
                            params.temperature_downsample,
                        ) > 0.
                        # -- determine how much to transfer between all sources
                        #    and sinks
                        num_sinks_unfrozen = sinks_unfrozen.sum()
                        num_sources_unfrozen = sources_unfrozen.sum()
                        total_source_take = jnp.minimum(
                            num_sinks_unfrozen * params.water_sink_flow,
                            num_sources_unfrozen * params.water_source_flow,
                        ).astype(float_dtype)
                        sink_take = jnp.where(
                            sinks_unfrozen,
                            total_source_take / num_sinks_unfrozen,
                            0.,
                        )
                        active_sources = sources_unfrozen
                        num_active_sources = num_sources_unfrozen
                    else:
                        total_source_take = jnp.minimum(
                            num_water_sinks * params.water_sink_flow,
                            num_water_sources * params.water_source_flow,
                        )
                        sink_take = jnp.full(
                            (num_water_sinks,),
                            total_source_take / num_water_sinks,
                            dtype=float_dtype,
                        )
                        active_sources = jnp.ones(
                            (num_water_sources,),
                            dtype=jnp.bool,
                        )
                        num_active_sources = num_water_sources
                    # -- take the water from the sinks
                    water, sunk_water = take_from_grid_locations(
                        water,
                        state.water_sink_locations,
                        sink_take,
                        params.terrain_downsample,
                    )
                    # -- add the water to the sources
                    total_sunk_water = jnp.sum(sunk_water)
                    source_add = jnp.where(
                        active_sources,
                        total_sunk_water / num_active_sources,
                        0.,
                    )
                    water = add_to_grid_locations(
                        water,
                        state.water_source_locations,
                        source_add,
                        params.terrain_downsample,
                    )
                
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
                    params.erosion_ratio,
                )
                erosion = reset_erosion_status(rock, old_rock, erosion)
                # - recompute normals
                state = state.replace(rock=rock, erosion=erosion)
            
            # compute altitude
            normalized_altitude = compute_normalized_altitude(state)
            
            # wind
            if params.include_wind:
                key, wind_key = jrng.split(key)
                wind = wind_system.step(wind_key, state.wind)
                state = state.replace(wind=wind)
            
            # rain
            if params.include_rain:
                # compute evaporation (water -> moisture)
                # - use the temperature to figure out how much is evaporating
                local_temperature = grid_sum_to_mean(
                    state.temperature, params.temperature_downsample)
                evaporation = (
                    (local_temperature - params.min_evaporation_temp) /
                    evaporation_temp_range
                )
                evaporation = jnp.clip(evaporation, min=0., max=1.)
                evaporation *= params.evaporation_rate
                # - turn off evaporation wherever it's raining
                evaporation = scale_grid(
                    evaporation, jnp.logical_not(state.raining))
                #evaporation *= (
                #    params.evaporation_rate * (params.rain_downsample**2))
                evaporation = grid_mean_to_sum(
                    evaporation, params.temperature_downsample)
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
                #altitude_rain_scale = (
                #    1. - normalized_altitude + params.rain_altitude_scale)
                #altitude_rain_scale = (
                #    normalized_altitude +
                #    (1 - normalized_altitude) * params.rain_altitude_scale
                #)
                altitude_rain_scale = (
                    params.rain_altitude_scale + 
                    normalized_altitude * (1. - params.rain_altitude_scale)
                )
                # - where should it start raining
                moisture_start_raining = (
                    params.moisture_start_raining *
                    altitude_rain_scale *
                    params.rain_downsample**2
                )
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
            
            # light
            if params.include_light:
                # - seasonal
                light_strength = get_day_light_strength(t)
                # - mask the terrain normals based on standing water which
                #   is approximated to being flat everywhere in order to avoid
                #   recomputing normals at each time step
                standing_water_light = subsample_grid(
                    standing_water, *state.light.shape[:2], preserve_mass=False)
                terrain_normals = jnp.where(
                    standing_water_light[..., None],
                    jnp.array([0.,0.,1.], dtype=float_dtype),
                    state.rock_normals,
                )
                # - light step
                light = light_step(
                    params.steps_per_day,
                    terrain_normals, 
                    light_strength,
                    light_length,
                    t,
                    params.night_effect
                )
                light = grid_mean_to_sum(light, params.light_downsample)
            
                # - reduce light based on shade from clouds and rain
                if params.include_rain:
                    clouds = scale_grid(moisture, 1./moisture_start_raining)
                    clouds = jnp.clip(clouds, min=0., max=1.)
                    shade = 1. - (
                        clouds*params.cloud_shade + raining*params.rain_shade)
                    light = scale_grid(light, shade)
                
                state = state.replace(light=light)
            
            # temperature
            if params.include_temperature:
                # - apply gas dynamics
                key, temperature_key = jrng.split(key)
                temperature = temperature_system.step(
                    temperature_key, state.temperature, wind=wind)
                
                # - update the temperature based on the incoming light
                # -- compute the temperature blend
                temperature_alpha = jnp.where(
                    standing_water_light,
                    params.water_thermal_mass,
                    params.ground_thermal_mass,
                )
                # -- compute the target temperature
                # --- the temperature_baseline represents the temperature a
                #     particular altitude should settle to with no incoming
                #     light
                light_shape = downsample_grid_shape(
                    *params.world_size, params.light_downsample)
                light_alpha = subsample_grid(normalized_altitude, *light_shape)
                temperature_baseline = grid_mean_to_sum(
                    light_alpha * params.mountain_temperature_baseline +
                    (1. - light_alpha) * params.sea_level_temperature_baseline,
                    params.terrain_downsample,
                )
                # --- the heat_absorption represents how quickly the surface
                #     is heated by incoming light
                heat_absorption = jnp.where(
                    standing_water_light,
                    params.water_heat_absorption,
                    params.ground_heat_absorption,
                )
                # --- c is a correction factor due to the light not being full
                #     strength all day
                c = jnp.array((4./jnp.pi), dtype=float_dtype)
                target_temperature = add_grids(
                    temperature_baseline,
                    c * state.light * heat_absorption,
                )
                
                # -- interpolate between the current and target temperature
                temperature = interpolate_grids(
                    temperature, target_temperature, temperature_alpha)
                
                # - update state
                state = state.replace(temperature=temperature)
            
            return state
        
        def render_rgb(
            state,
            shape,
            spot_x=None,
            spot_color=None,
            use_light=True,
        ):
            h, w = shape
            assert params.world_size[0] % h == 0
            assert params.world_size[1] % w == 0
            
            # make everything rock colored
            rgb = jnp.full((h,w,3), ROCK_COLOR, dtype=float_dtype)
            
            # overlay water and ice
            if params.include_temperature:
                th, tw = state.temperature.shape
                water_color = jnp.where(
                    state.temperature[..., None] <= 0.,
                    ICE_COLOR,
                    WATER_COLOR,
                )
                wh, ww = state.water.shape
                water_color = upsample_grid(
                    water_color, wh, ww, preserve_mass=False)
            else:
                water_color = jnp.full((*state.water.shape, 3), WATER_COLOR)
            standing_water_threshold = (
                params.min_standing_water * (params.terrain_downsample**2))
            standing_water = (
                state.water[..., None] >= standing_water_threshold)
            rgb = interpolate_grids(
                rgb, water_color, ~standing_water, preserve_mass=False)
            
            # apply the energy and biomass tint
            energy = grid_sum_to_mean(
                state.energy, params.resource_downsample)
            energy = jnp.clip(energy, min=0, max=1)
            biomass = grid_sum_to_mean(
                state.biomass, params.resource_downsample)
            biomass = jnp.clip(biomass, min=0, max=1)
            biomass_and_energy = jnp.minimum(energy, biomass)
            just_energy = energy - biomass_and_energy
            just_biomass = biomass - biomass_and_energy
            rgb = add_grids(
                rgb,
                biomass_and_energy[..., None] * BIOMASS_AND_ENERGY_TINT,
                preserve_mass=False,
            )
            rgb = add_grids(
                rgb,
                just_energy[..., None] * ENERGY_TINT,
                preserve_mass=False,
            )
            rgb = add_grids(
                rgb,
                just_biomass[..., None] * BIOMASS_TINT,
                preserve_mass=False,
            )
            
            if spot_x is not None:
                rgb_downsample = params.world_size[0] // h
                rgb = write_grid_locations(
                    rgb, spot_x, spot_color, rgb_downsample)
            
            if use_light:
                light = grid_sum_to_mean(state.light, params.light_downsample)
                rgb = scale_grid(rgb, light[..., None])
            
            return rgb
        
        def render_temperature(state, shape):
            if params.include_temperature:
                temperature = state.temperature[..., None]
                temperature = grid_sum_to_mean(
                    temperature, params.temperature_downsample)
                hot = jnp.array([0.5, 0., 0.], dtype=float_dtype)
                cold = jnp.array([0., 0., 0.5], dtype=float_dtype)
                rgb = jnp.where(
                    temperature >= 0.,
                    temperature * hot,
                    -temperature * cold,
                )
                rgb = set_grid_shape(rgb, *shape, preserve_mass=False)
            else:
                rgb = jnp.zeros((*shape, 3), dtype=float_dtype)
            
            return rgb
        
        def render_weather(state, shape):
            if params.include_rain:
                moisture = state.moisture[..., None]
                moisture = grid_sum_to_mean(moisture, params.rain_downsample)
                normalized_moisture = moisture / params.moisture_start_raining
                normalized_moisture = jnp.clip(normalized_moisture, max=1.)
                rain_color = jnp.array([0.25, 0.25, 1.], dtype=float_dtype)
                rgb = jnp.where(
                    state.raining[..., None], rain_color, normalized_moisture)
                rgb = set_grid_shape(rgb, *shape, preserve_mass=False)
            else:
                rgb = jnp.zeros((*shape, 3), dtype=float_dtype)
            
            return rgb
        
        def render_altitude(state, shape):
            normalized_altitude = compute_normalized_altitude(state)
            normalized_altitude = set_grid_shape(
                normalized_altitude, *shape, preserve_mass=False)
            rgb = (
                normalized_altitude[..., None] *
                jnp.array([1., 1., 1.], dtype=float_dtype)
            )
            rgb = set_grid_shape(rgb, *shape, preserve_mass=False)
            
            return rgb
    
    return Landscape
