from typing import Tuple, Any

import jax.random as jrng
import jax.numpy as jnp
import jax

from mechagogue.static import static_data, static_functions

from dirt.constants import DEFAULT_FLOAT_DTYPE
from dirt.distribution.ou import ou_process
from dirt.gridworld2d.blocks import align_dense_to_blocks, blocks_to_dense
from dirt.gridworld2d.geology import fractal_noise
from dirt.gridworld2d.water import flow_step_twodir
from dirt.gridworld2d.erosion import simulate_erosion_step, reset_erosion_status
from dirt.gridworld2d.gas import gas

@static_data
class WeatherParams:
    
    # general
    world_size: Tuple[int,int] = (1024,1024)
    
    max_effective_altitude = 100.
    min_effective_water = 0.05
    
    # wind
    include_wind: bool = True
    wind_std: float = 3
    wind_reversion: float = 0.1
    wind_bias: Tuple[float,float] = (0.,0.)
    
    # temperature
    include_temperature: bool = True
    temperature_downsample : int = 8
    # notes on temperature:
    # DIRT temperatures are approximately 1 unit = 20c
    # The defaults here mean that:
    #  - the highest mountain peak with no sunlight will stabilize at -40c
    #  - the highest mountain peak with full sunlight will stabilize at 20c
    #  - sea level with no sunlight will stabilize at -20c
    #  - sea level with full sunlight will stabilize at 40c
    #  - sea level over water will stabilize at 30c
    initial_temperature: float = 1.
    sea_level_temperature_baseline: float = -1.
    mountain_temperature_baseline: float = -2.
    ground_heat_absorption: float = 3.
    water_heat_absorption: float = 2.
    ground_thermal_mass: float = 0.995
    water_thermal_mass: float = 0.999
    temperature_diffusion_radius: int = 1
    temperature_diffusion_strength: float = 1.
    
    # rain
    include_rain: bool = True
    rain_downsample: int = 8
    
    # - moisture
    initial_moisture: float = 0.
    evaporation_rate: float = 0.001
    min_evaporation_temp: float = 0.1
    max_evaporation_temp: float = 2.
    
    # - rain
    rain_per_step: float = 0.005
    moisture_start_raining: float = 0.2
    moisture_stop_raining: float = 0.05
    rain_altitude_scale: float = 0.25
    
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
    

def make_weather(
    params: WeatherParams,
    float_dtype: Any = DEFAULT_FLOAT_DTYPE,
):
    
    # setup the wind system
    if params.include_wind:
        wind_system = ou_process(
            params.wind_std * jnp.sqrt(2*params.wind_reversion),
            params.wind_reversion,
            jnp.array(params.wind_bias, dtype=float_dtype),
            dtype=float_dtype,
        )
        max_wind = int(params.wind_std * 3)
    else:
        max_wind = 0
    
    # setup the temperature
    if params.include_temperature:
        temperature_system = gas(
            params.world_size,
            downsample=params.temperature_downsample,
            initial_value=params.initial_temperature,
            diffusion_radius=params.temperature_diffusion_radius,
            diffusion_strength=params.temperature_diffusion_strength,
            boundary='edge',
            max_wind=max_wind,
            float_dtype=float_dtype,
        )
    
    # setup the moisture system
    if params.include_rain:
        evaporation_temp_range = (
            params.max_evaporation_temp - params.min_evaporation_temp)
        moisture_system = gas(
            params.world_size,
            downsample=params.rain_downsample,
            initial_value=params.initial_moisture,
            diffusion_radius=params.moisture_diffusion_radius,
            diffusion_strength=params.moisture_diffusion_strength,
            boundary='collect',
            max_wind=max_wind,
            float_dtype=float_dtype,
        )
    
    # setup the smell system
    if params.include_smell:
        smell_system = gas(
            params.world_size,
            downsample=params.smell_downsample,
            cell_shape=(params.smell_channels,),
            initial_value=0.,
            diffusion_radius=params.smell_diffusion_radius,
            diffusion_strength=params.smell_diffusion_strength,
            dissipation=params.smell_dissipation,
            boundary='clip',
            max_wind=max_wind,
            float_dtype=float_dtype,
        )
    
    # setup the audio system
    if params.include_audio:
        audio_system = gas(
            params.world_size,
            downsample=params.audio_downsample,
            cell_shape=params(audio_channels,),
            initial_value=0.,
            diffusion_radius=params.audio_diffusion_radius,
            diffusion_strength=params.audio_diffusion_strength,
            dissipation=1.,
            boundary='clip',
            include_wind=False,
            float_dtype=float_dtype,
        )
    
    def temperature_step(
        key,
        light,
        water,
        normalized_altitude,
        wind,
        temperature,
    ):
        # make sure light, water and normalized altitude are the same shape
        lh, lw = light.shape
        wh, ww = water.shape
        ah, aw = normalized_altitude.shape
        assert lh == wh and lh == ah and lw == ww and lw == aw
         
        # diffuse the temperature
        next_temperature = temperature_system.step(key, temperature)
        
        # compute the standing water
        standing_water = water > params.min_effective_water
        
        # compute the temperature blend
        temperature_alpha = jnp.where(
            standing_water,
            params.water_thermal_mass,
            params.ground_thermal_mass,
        )
        
        # compute the target_temperature
        temperature_baseline = (
            normalized_altitude * params.mountain_temperature_baseline +
            (1. - normalized_altitude) * params.sea_level_temperature_baseline
        )
        heat_absorption = jnp.where(
            standing_water,
            params.water_heat_absorption,
            params.ground_heat_absorption,
        )
        # - this c factor corrects for the fact that the light is not full
        #   strength all day long
        c = 4./jnp.pi
        target_temperature = temperature_baseline + c * light * heat_absorption
        
        # align blocks
        next_temperature_b, target_temperature_b = align_dense_to_blocks(
            next_temperature, target_temperature)
        
        # incorporte the target temperature with an exponential moving average
        next_temperature_b = (
            next_temperature_b * temperature_alpha +
            target_temperature_b * (1. - temperature_alpha)
        )
        next_temperature = blocks_to_dense(next_temperature_b)
        
        return next_temperature
    
    def evaporate_step(
        water: jnp.ndarray,
        temperature: jnp.ndarray,
        moisture: jnp.ndarray,
        rain: jnp.ndarray,
    ) -> jnp.ndarray:
        '''
        evaporate a certain amount of water into the atmosphere
            Here I made it little similar to real life,
            where the evaporate only depends on the temperature/humidity of the
            certain place based on simple search of the Internet
        
        moisture: water evaporated in the air in each area of the
            world
        rain: 1 for raining and 0 not
        '''
        evaporation_ammount = (
            (temperature - params.min_evaporation_temp) /
            evaporation_temp_range
        )
        evaporation_ammount = jnp.clip(evaporation_ammount, min=0., max=1.)
        evaporation_ammount = evaporation_ammount
        evaporation = jnp.where(
            rain, 0, params.evaporation_rate * evaporation_ammount)
        evaporation = jnp.minimum(evaporation, water)
        next_water = water - evaporation
        next_moisture = moisture + evaporation
        
        return next_water, next_moisture

    def rain_step(
        normalized_altitude: jnp.ndarray,
        water: jnp.ndarray,
        moisture: jnp.ndarray,
        rain: jnp.ndarray,
        wind: jnp.ndarray,
    ) -> jnp.ndarray:
        '''
        Rain when the water in the atmosphere surpasses a certain value
        When it's raining, evaporation stops.

        air_limit: how much water the atmosphete can hold
        '''
        rain_amount = jnp.where(
            rain, params.rain_per_step, 0)
        rain_amount = jnp.clip(rain_amount, max=moisture)
        next_water = water + rain_amount
        next_moisture = moisture - rain_amount
        altitude_rain_scale = (
            1. - normalized_altitude + params.rain_altitude_scale)
        moisture_start_raining = (
            params.moisture_start_raining * altitude_rain_scale)
        moisture_stop_raining = (
            params.moisture_stop_raining * altitude_rain_scale)
        start_raining = moisture > moisture_start_raining
        stop_raining = moisture <= moisture_stop_raining
        next_rain = (rain | start_raining) & ~stop_raining
        
        # smooth out the features
        # turn off rain where there are less than three neighbors also raining
        # turn on rain  where there are more than six neighbors also raining
        kernel = jnp.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ], dtype=jnp.int8).reshape((3,3,1,1))
        raining_neighbors = jax.lax.conv_general_dilated(
            next_rain.astype(jnp.int8)[None,...,None],
            kernel,
            window_strides=(1,1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )[0,...,0]
        next_rain = (next_rain & raining_neighbors > 3) | raining_neighbors > 6
        
        next_rain = jnp.roll(next_rain, shift=wind, axis=(0,1))
        
        return next_water, next_moisture, next_rain
    
    @static_functions
    class Weather:
        @static_data
        class WeatherState:
            wind : jnp.ndarray
            temperature : jnp.ndarray
            moisture : jnp.ndarray
            rain : jnp.ndarray
            smell : jnp.ndarray
            audio : jnp.ndarray
        
        def init(key):
            # initialize the wind
            if params.include_wind:
                key, wind_key = jrng.split(key)
                wind = init_wind(key)
            else:
                wind = jnp.zeros((2,))
            
            # initialize temperature
            if params.include_temperature:
                temperature = jnp.full(
                    params.world_size,
                    params.initial_temperature,
                    dtype=float_dtype,
                )
            else:
                temperature = jnp.full(
                    (), params.initial_temperature, dtype=float_dtype)
            
            # initialize rain and moisture
            if params.include_rain:
                moisture = init_moisture()
                rain = jnp.zeros(params.world_size, dtype=jnp.bool)
            else:
                moisture = jnp.zeros((), dtype=float_dtype)
                rain = jnp.zeros((), dtype=jnp.bool)
            
            return WeatherState(wind, temperature, moisture, rain)
        
        def step(
            key,
            water: jnp.ndarray,
            #temperature: jnp.ndarray,
            #moisture: jnp.ndarray, 
            #rain: jnp.ndarray,
            #wind: jnp.ndarray,
            normalized_altitude: jnp.ndarray,
            light: jnp.ndarray,
            weather_state,
        ) -> jnp.ndarray:
            
            # change wind direction
            if params.include_wind:
                key, wind_key = jrng.split(key)
                wind = step_wind(wind_key, weather_state.wind)
            else:
                wind = None
            
            # rain
            if params.include_rain:
                # evaporate
                water, moisture = evaporate_step(
                    water, temperature, moisture, rain)
                
                # make it rain
                water, moisture, rain = rain_step(
                    normalized_altitude, water, moisture, rain, wind)
                
                # blow the moisture around
                key, moisture_key = jrng.split(key)
                moisture = moisture_step(moisture_key, moisture, wind)
            
            # update the temperature
            if params.include_temperature:
                key, temperature_key = jrng.split(key)
                temperature = temperature_step(
                    temperature_key,
                    light,
                    water,
                    normalized_altitude,
                    temperature,
                )
            #state =State(temperature, moisture, rain, wind, discrete_wind) 
            return water, WeatherState(temperature, moisture, rain, wind)
    
    return Weather
