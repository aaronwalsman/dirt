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

# temperature units:
# 0: -50C
# 1: 25C

@static_dataclass
class WeatherParams:
    # general
    world_size: Tuple[int,int] = (32, 32)
    
    # temperature
    min_temperature: float = -1.
    initial_temperature: float = 1.
    ground_heat_absorption: float = 0.01
    water_heat_absorption: float = 0.005
    moisture_heat_exchange_factor: float = 0.5
    elevation_: 
    
    # moisture
    initial_moisture: float = 0.
    evaporation_rate: float = 0.01
    min_evaporation_temp: float = 0.1
    max_evaporation_temp: float = 2.
    moisture_std: float = 1
    moisture_mix: float = 1.
    
    # rain
    rain_per_step: float = 0.32
    moisture_start_raining: float = 1.0
    moisture_stop_raining: float = 0.
    
    # wind
    wind_std: float = 1.
    wind_reversion: float = 0.001
    wind_bias: Tuple[float,float] | jnp.ndarray = (0.,0.)

def weather(
    params: WeatherParams,
    step_size: float = 1.,
    float_dtype: Any = DEFAULT_FLOAT_DTYPE,
):
    
    evaporation_temp_range = (
        params.max_evaporation_temp - params.min_evaporation_temp)
    
    init_wind, step_wind = ou_process(
        params.wind_std,
        params.wind_reversion,
        jnp.array(params.wind_bias, dtype=float_dtype),
        dtype=float_dtype,
    )
    
    moisture_diffusion_step = gas(
        params.moisture_std,
        params.moisture_mix,
        boundary='wrap',
        float_dtype=float_dtype,
    )
    
    def temperature_step(
        water,
        temperature,
        moisture,
        rain,
        altitude,
        light,
    ):
        
        
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
        water: jnp.ndarray,
        moisture: jnp.ndarray,
        rain: jnp.ndarray,
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
        next_rain = jnp.where(
            (rain == 0) & (moisture > params.moisture_start_raining),
            True,
            jnp.where((rain) & (moisture < params.moisture_stop_raining), False, rain)
        )
        
        return next_water, next_moisture, next_rain
    
    def init(key):
        temperature = jnp.full(
            params.world_size, params.initial_temperature, dtype=float_dtype)
        moisture = jnp.full(
            params.world_size, params.initial_moisture, dtype=float_dtype)
        rain = jnp.zeros(params.world_size, dtype=jnp.bool)
        key, wind_key = jrng.split(key)
        wind = init_wind(key)
        
        return temperature, moisture, rain, wind 
    
    def step(
        key,
        water: jnp.ndarray,
        temperature: jnp.ndarray,
        moisture: jnp.ndarray, 
        rain: jnp.ndarray,
        wind: jnp.ndarray,
        terrain: jnp.ndarray,
        light: jnp.ndarray,
    ) -> jnp.ndarray:
        '''
        Put evaporation and rain together to make the naive
        full weather system
        '''
        # update the temperature
        altitude = terrain + water
        temperature = temperature_step(
            temperature,
            altitutde,
            light,
        )
        
        # evaporate
        water, moisture = evaporate_step(water, temperature, moisture, rain)
        
        # change wind direction
        key, wind_key = jrng.split(key)
        wind = step_wind(wind_key, wind, step_size=step_size)
        
        # blow the moisture around
        moisture = moisture_diffusion_step(moisture, wind)
        
        # make it rain
        water, moisture, rain = rain_step(water, moisture, rain)
        
        return water, moisture, rain, wind
    
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

