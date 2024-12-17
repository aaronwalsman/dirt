from geology import Fractal_Noise
from water import flow_step_twodir
from erosion import simulate_erosion_step, reset_erosion_status
import jax.random as jrng
import jax.numpy as jnp
import jax

from typing import Tuple, Optional, Union

'''
To make the whole world little more interesting...
Why not add the weather system?
Here is a naive version, only adding the evaporation and rain drops
'''

def evaporate_step(
    water: jnp.ndarray,
    temperature: jnp.ndarray,
    current_evaporation: jnp.ndarray,
    rain_status: jnp.ndarray,
    evaporate_rate: float
) -> jnp.ndarray:
    '''
    evaporate a certain amount of water into the atmosphere
    Here I made it little similar to real life,
    where the evaporate only depends on the temperature/humidity of the certain place based on
        simple search of the Internet

    current_evaporation: water evaporated in the air in each area of the world
    rain_status: 1 for raining and 0 not
    evaporate_rate: how fast water is evaporated to the atmosphere
    '''
    temperature = (temperature - jnp.min(temperature)) / (jnp.max(temperature) - jnp.min(temperature))
    evaporation_state = jnp.where(rain_status == 0, evaporate_rate * temperature * jnp.mean(water), 0)
    # evaporation_state = jnp.where(rain_status == 0, evaporate_rate * temperature * jnp.mean(water) * (jnp.max(current_evaporation)-current_evaporation), 0)
    evaporation_state = jnp.minimum(evaporation_state, water)
    new_water = water - evaporation_state
    new_evaporation = current_evaporation + evaporation_state
    return new_water, new_evaporation

def rain_step(
    current_evaporation: jnp.ndarray,
    water: jnp.ndarray,
    air_up_limit: float,
    air_down_limit: float,
    rain_status: jnp.ndarray,
    rain: float
) -> jnp.ndarray:
    '''
    Rain when the water in the atmosphere surpasses a certain value
    When it's raining, evaporation stops.

    air_limit: how much water the atmosphete can hold
    rain: amount of water from air to the ground
    '''
    rain_amount = jnp.where(rain_status, rain, 0)
    new_water = water + rain_amount
    new_evaporation = current_evaporation - rain_amount
    new_rain_status = jnp.where(
        (rain_status == 0) & (current_evaporation > air_up_limit), 1,
        jnp.where((rain_status == 1) & (current_evaporation < air_down_limit), 0, rain_status)
    )
    return new_water, new_evaporation, new_rain_status

def weather_step(
    water: jnp.ndarray,
    temperature: jnp.ndarray,
    current_evaporation: jnp.ndarray, 
    rain_status: int,
    evaporate_rate: float,
    air_up_limit: float,
    air_down_limit: float,
    rain: float
) -> jnp.ndarray:
    '''
    Put evaporation and rain together to make the naive
    full weather system
    '''
    water, current_evaporation = evaporate_step(
        water, temperature, current_evaporation, rain_status, evaporate_rate
    )
    water, current_evaporation, rain_status = rain_step(current_evaporation, water, air_up_limit, air_down_limit, rain_status, rain)
    return water, current_evaporation, rain_status

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
    terrain = Fractal_Noise(world_size=world_size, octaves = 6, persistence = 0.5, lacunarity = 2.0, key = key)
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

