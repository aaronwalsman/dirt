from typing import Tuple, Any

from mechagogue.static_dataclass import static_dataclass

from dirt.gridworld2d.geology import fractal_noise
from dirt.gridworld2d.water import flow_step_twodir
from dirt.gridworld2d.erosion import simulate_erosion_step, reset_erosion_status
from dirt.gridworld2d.gas import gas
from dirt.distribution.ou import ou_process
from dirt.constants import DEFAULT_FLOAT_DTYPE
import jax.random as jrng
import jax.numpy as jnp
import jax

'''
To make the whole world little more interesting...
Why not add the weather system?
Here is a naive version, only adding the evaporation and rain drops
'''

@static_dataclass
class WeatherParams:
    
    # general
    world_size: Tuple[int,int] = (32, 32)
    max_effective_altitude = 100.
    min_effective_water = 0.05
    
    # temperature
    include_temperature: bool = True
    # notes on temperature:
    # DIRT temperatures are approximately 1 unit = 20c
    # The defaults here mean that:
    #  - the highest mountain peak with no sunlight will stabilize at -40c
    #  - the highest mountain peak with full sunlight will stabilize at 20c
    #  - sea level with no sunlight will stabilize at -20c
    #  - sea level with full sunlight will stabilize at 40c
    #  - sea level over water will stabilize at 30c
    # need to adjust due to expected sun per day
    initial_temperature: float = 1.
    sea_level_temperature_baseline: float = -1.
    mountain_temperature_baseline: float = -2.
    ground_heat_absorption: float = 3.
    water_heat_absorption: float = 2.
    ground_thermal_mass: float = 0.995
    water_thermal_mass: float = 0.999
    temperature_std: float = 1.
    
    # rain
    include_rain: bool = True
    
    # - moisture
    initial_moisture: float = 0.
    evaporation_rate: float = 0.001
    min_evaporation_temp: float = 0.1
    max_evaporation_temp: float = 2.
    moisture_std: float = 0.333
    
    # - rain
    rain_per_step: float = 0.01
    moisture_start_raining: float = 0.2
    moisture_stop_raining: float = 0.05
    rain_altitude_scale: float = 0.25
    
    # wind
    include_wind: bool = True
    wind_std: float = 3
    wind_reversion: float = 0.1
    wind_bias: Tuple[float,float] = (0.,0.)

def weather(
    params: WeatherParams,
    step_size: float = 1.,
    float_dtype: Any = DEFAULT_FLOAT_DTYPE,
):
    
    if params.include_rain:
        evaporation_temp_range = (
            params.max_evaporation_temp - params.min_evaporation_temp)
    
    # setup the wind
    if params.include_wind:
        init_wind, step_wind_ou = ou_process(
            params.wind_std * jnp.sqrt(2*params.wind_reversion),
            params.wind_reversion,
            jnp.array(params.wind_bias, dtype=float_dtype),
            dtype=float_dtype,
        )
        
        def step_wind(key, wind):
            ou_key, round_key = jrng.split(key)
            wind = step_wind_ou(ou_key, wind, step_size=step_size)
            wind = wind * step_size
            wind_floor = jnp.floor(wind)
            wind_ceil = jnp.ceil(wind)
            p_ceil = wind - wind_floor
            round_direction = jrng.uniform(round_key, wind.shape)
            discrete_wind = jnp.where(
                round_direction < p_ceil, wind_ceil, wind_floor)
            return wind, discrete_wind
    
    if params.include_temperature:
        (
            init_temperature,
            temperature_diffusion_step,
            read_temperature,
            _,
            _,
        ) = gas(
            params.world_size,
            diffusion_std=params.temperature_std,
            boundary='wrap',
            include_wind=False,
            step_size=step_size,
            float_dtype=float_dtype,
        )
    
    if params.include_rain:
        init_moisture, moisture_step, read_moisture, _, add_moisture = gas(
            params.world_size,
            diffusion_std=params.moisture_std,
            boundary='wrap',
            step_size=step_size,
            float_dtype=float_dtype,
        )
    
    #init_smell, smell_step, read_smell, _, add_smell = gas(
    #    ...
    #)
    
    #init_audio, audio_step, read_audio, _, add_audio = gas(
    #    ...
    #)
    
    def temperature_step(
        key,
        water,
        temperature,
        normalized_altitude,
        light,
    ):
        # diffuse the temperature
        next_temperature = temperature_diffusion_step(key, temperature)
        
        # compute the standing water
        standing_water = water > params.min_effective_water
        
        # compute the temperature blend
        temperature_alpha = jnp.where(
            standing_water,
            params.water_thermal_mass,
            params.ground_thermal_mass,
        ) ** step_size
        
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
        
        # incorporte the target temperature with an exponential moving average
        next_temperature = (
            next_temperature * temperature_alpha +
            target_temperature * (1. - temperature_alpha)
        )
        
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
        evaporation_ammount = evaporation_ammount * step_size
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
        rain_amount = jnp.where(rain, params.rain_per_step * step_size, 0)
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
    
    def init(key):
        if params.include_wind:
            key, wind_key = jrng.split(key)
            wind = init_wind(key)
        else:
            wind = jnp.zeros((2,))
        
        if params.include_temperature:
            temperature = jnp.full(
                params.world_size,
                params.initial_temperature,
                dtype=float_dtype,
            )
        else:
            temperature = jnp.full(
                (), params.initial_temperature, dtype=float_dtype)
        
        if params.include_rain:
            moisture = init_moisture()
            rain = jnp.zeros(params.world_size, dtype=jnp.bool)
        else:
            moisture = jnp.zeros((), dtype=float_dtype)
            rain = jnp.zeros((), dtype=jnp.bool)
        
        return temperature, moisture, rain, wind 
    
    def step(
        key,
        water: jnp.ndarray,
        temperature: jnp.ndarray,
        moisture: jnp.ndarray, 
        rain: jnp.ndarray,
        wind: jnp.ndarray,
        normalized_altitude: jnp.ndarray,
        light: jnp.ndarray,
    ) -> jnp.ndarray:
        
        # change wind direction
        if params.include_wind:
            key, wind_key = jrng.split(key)
            wind, discrete_wind = step_wind(wind_key, wind)
        else:
            discrete_wind = jnp.zeros((2,), dtype=jnp.int32)
        
        # rain
        if params.include_rain:
            # evaporate
            water, moisture = evaporate_step(water, temperature, moisture, rain)
            
            # make it rain
            water, moisture, rain = rain_step(
                normalized_altitude, water, moisture, rain, discrete_wind)
            
            # blow the moisture around
            key, moisture_key = jrng.split(key)
            moisture = moisture_step(moisture_key, moisture, discrete_wind)
        
        # update the temperature
        if params.include_temperature:
            key, temperature_key = jrng.split(key)
            temperature = temperature_step(
                temperature_key, water, temperature, normalized_altitude, light)
        
        return water, temperature, moisture, rain, wind, discrete_wind
    
    return init, step

def simulate_weather(
    terrain: jnp.ndarray,
    water: jnp.ndarray,
    temperature: jnp.ndarray,
    time: int,
    erosion_initial: jnp.ndarray,
    initial_rain_status: jnp.ndarray,
    initial_evaporation: jnp.ndarray,
    evaporate_rate: float,
    air_up_limit: float,
    air_down_limit: float,
    rain: float,
    flow_rate: float,
    erosion_ratio: float,
    erosion_endurance: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    '''
    Simulate the whole system to see how it goes!

    The order of natural weather will be: 
        1. water flow
        2. erosion
        3. evaporate/rain_fall
            3.1 evaporate
            3.2 rain_fall
    '''
    def step_fn(carry, _):
        terrain, water, current_evaporation, rain_status, current_erosion = carry
        
        # 1. Water flow
        new_water_1 = flow_step_twodir(terrain, water, flow_rate)
        
        # 2. Erosion
        new_terrain = simulate_erosion_step(terrain, water, current_erosion, flow_rate, erosion_endurance, erosion_ratio)
        current_erosion = reset_erosion_status(new_terrain, terrain, current_erosion)
        
        # 3. Weather system
        new_water, new_evaporation, rain_status = weather_step(
            new_water_1, temperature, current_evaporation, rain_status, evaporate_rate, air_up_limit, air_down_limit, rain)
        
        return (new_terrain, new_water, new_evaporation, rain_status, current_erosion), None
    
    initial_state = (terrain, water, initial_evaporation, initial_rain_status, erosion_initial)
    final_state, _ = jax.lax.scan(step_fn, initial_state, None, length=time)

    final_terrain, final_water, final_evaporation, rain_status, _ = final_state
    return final_terrain, final_water, final_evaporation, rain_status

if __name__ == '__main__':
    # Here for test purpose, we simply make the temperature to be positively correlated to
    # the height of the terrain, simulating that there is always a Sun in the system if no rain
    key = jrng.PRNGKey(1022)
    world_size = (256,256)
    erosion_initial = jnp.zeros(world_size)
    water_initial = 0.5
    evaporation_initial = 0
    rain_initial = 0
    flow_rate = 0.45
    terrain = fractal_noise(key=key, world_size=world_size, octaves = 6, persistence = 0.5, lacunarity = 2.0)
    water = jnp.full(world_size, water_initial)
    temperature_naive = (terrain - jnp.min(terrain)) / (jnp.max(terrain) - jnp.min(terrain))
    time = 500
    rain_initial = jnp.full(world_size, rain_initial)
    evaporation_initial = jnp.full(world_size, evaporation_initial)
    evaporate_rate = 0.01
    air_up_limit = 0.2
    air_down_limit= 0.1
    rain = 0.08
    erosion_endurance = 0.05
    erosion_ratio = 0.01

    final_terrain, final_water, left_evaporation, final_rain_status = simulate_weather(
        terrain, water, temperature_naive, time, 
        erosion_initial, rain_initial, evaporation_initial, 
        evaporate_rate, air_up_limit, air_down_limit, rain,
        flow_rate, erosion_ratio, erosion_endurance 
    )
    total_water = final_water + left_evaporation
    print(terrain.sum()) # -349.00833
    print(final_terrain.sum()) # -349.00833
    print(water.sum()) # 32768.0
    print(total_water.sum()) # 32767.998
    print(final_terrain == terrain)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Final Terrain")
    plt.imshow(final_terrain, cmap="terrain")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Final Water")
    plt.imshow(final_water, cmap="Blues")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Final Evaporation")
    plt.imshow(left_evaporation, cmap="YlGnBu")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

