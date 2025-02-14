from geology import fractal_noise
from water import flow_step_twodir
from erosion import simulate_erosion_step, reset_erosion_status
from naive_weather_system import weather_step
from climate_pattern_day import temperature_step, light_step, get_day_status
import jax.random as jrng
import jax.numpy as jnp
import jax

from typing import Tuple, Optional, Union

'''
Outside of the day!

We won't make a clear cut between the four seasons to match the real life better,
but make season an interpretation for the temperature change, modeled by:

T = Light Strength(LS) + Light Time(LT) (+ \epsilon)

where the LS and LT are determined by the Earth's orbital revolution, and LT is also determined by the rain status of the region
\epsilon is the perturbation to capture some unexplained weather phenomena

One round roughly consists of 360 days
'''

def get_day_light_length(
    time: int
) -> float:
    '''
    Modeled as a triangular function, where the average day_light_length would be 12
    With summer to be 14 and winter to be 10
    '''
    return 12 + 2 * jnp.sin(time * jnp.pi / 180)

def get_day_light_strength(
    time: int
) -> float:
    '''
    Similar to the day light length
    We also model this into a triangular function, where the average day_light_strength would be 1
    With summer to be 1.2 and winter to be 0.8
    '''
    return 1 + 0.2 * jnp.sin(time * jnp.pi / 180)

def simulate_full_climate(
    terrain: jnp.ndarray,
    water: jnp.ndarray,
    temperature_initial: jnp.ndarray,
    time: int,
    erosion_initial: jnp.ndarray,
    initial_rain_status: jnp.ndarray,
    initial_evaporation: jnp.ndarray,
    day_status_initial: jnp.ndarray,
    light_intensity_initial: jnp.ndarray,
    evaporate_rate: float,
    air_up_limit: float,
    air_down_limit: float,
    rain: float,
    flow_rate: float,
    erosion_ratio: float,
    erosion_endurance: float, 
    night_effect: float, 
    water_effect: float, 
    rain_effect: float, 
    evaporation_effect: float
) -> jnp.ndarray:
    '''
    Simulate the light and temperature change in one full weather day
    '''
    def step_fn(carry, step_idx):
        terrain, water, current_evaporation, rain_status, current_erosion, day_status, light_intensity, current_temperature = carry
        
        current_time = step_idx
        # 1. Water flow
        water = flow_step_twodir(terrain, water, flow_rate)
        
        # 2. Erosion
        new_terrain, current_erosion = simulate_erosion_step(terrain, water, current_erosion, flow_rate, erosion_endurance, erosion_ratio)
        current_erosion = reset_erosion_status(new_terrain, terrain, current_erosion)

        # 3. light
        light_strength = get_day_light_strength(current_time)
        light_length = get_day_light_length(current_time)
        new_day_status = get_day_status(light_length, current_time)
        new_light_intensity = light_step(terrain, water, light_strength, light_length, current_time, night_effect) #Porblem of getting None

        # 4. Temperature
        new_temperature = temperature_step(current_time, water, current_temperature, rain_status, light_intensity, current_evaporation, light_length, night_effect, water_effect, rain_effect, evaporation_effect)
        
        # 5. Humidity
        new_water, new_evaporation, rain_status = weather_step(
            water, new_temperature, current_evaporation, rain_status, evaporate_rate, air_up_limit, air_down_limit, rain)
        
        return (new_terrain, new_water, new_evaporation, rain_status, current_erosion, new_day_status, new_light_intensity, new_temperature), None
    
    initial_state = (terrain, water, initial_evaporation, initial_rain_status, erosion_initial, day_status_initial,light_intensity_initial, temperature_initial)
    time_steps = jnp.arange(time)
    final_state, _ = jax.lax.scan(step_fn, initial_state, time_steps)

    final_terrain, final_water, final_evaporation, rain_status, final_erosion, final_day_status, final_light_intensity, final_temperature = final_state
    return final_terrain, final_water, final_evaporation, rain_status, final_erosion, final_day_status, final_light_intensity, final_temperature

if __name__ == '__main__':
    key = jrng.PRNGKey(1022)
    world_size = (256,256)
    erosion_initial = jnp.zeros(world_size)
    day_status_initial = 1
    water_initial = 0.5
    evaporation_initial_value = 0
    rain_initial_value = 0
    light_intensity_initial_value = 0.5
    temperature_initial_value = 1
    flow_rate = 0.45
    terrain = fractal_noise(key=key, world_size=world_size, octaves = 6, persistence = 0.5, lacunarity = 2.0)
    water = jnp.full(world_size, water_initial)
    time = 500
    rain_initial = jnp.full(world_size, rain_initial_value)
    evaporation_initial = jnp.full(world_size, evaporation_initial_value)
    light_intensity_initial = jnp.full(world_size, light_intensity_initial_value)
    temperature_initial = jnp.full(world_size, temperature_initial_value)
    evaporate_rate = 0.01
    air_up_limit = 0.2
    air_down_limit= 0.1
    rain = 0.08
    erosion_endurance = 0.05
    erosion_ratio = 0.01
    night_effect = 0.35 # Under this, the temperature system doesn't seem to explode
    water_effect = 0.5
    rain_effect = 0.1 
    evaporation_effect = 0.5

    final_terrain, final_water, left_evaporation, final_rain_status, final_erosion, final_day_status, final_light_intensity, final_temperature = simulate_full_climate(
        terrain, water, temperature_initial, time, 
        erosion_initial, rain_initial, evaporation_initial, day_status_initial, light_intensity_initial,  
        evaporate_rate, air_up_limit, air_down_limit, rain,
        flow_rate, erosion_ratio, erosion_endurance, night_effect, water_effect, rain_effect, evaporation_effect
    )
    total_water = final_water + left_evaporation
    print(terrain.sum()) # -349.00833
    print(final_terrain.sum()) # -349.00827
    print(water.sum()) # 32768.0
    print(total_water.sum()) # 32767.994
    print(final_terrain == terrain) # Erosion happening with this paramter

    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Final Terrain")
    plt.imshow(final_terrain, cmap="terrain")
    plt.colorbar()

    # plt.subplot(1, 3, 2)
    # plt.title("Final Water")
    # plt.imshow(final_water, cmap="Blues")
    # plt.colorbar()

    # plt.subplot(1, 3, 3)
    # plt.title("Final Evaporation")
    # plt.imshow(left_evaporation, cmap="YlGnBu")
    # plt.colorbar()

    # plt.subplot(1, 3, 1)
    # plt.title("Final Erosion")
    # plt.imshow(final_erosion, cmap="terrain")
    # plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Final light")
    plt.imshow(final_light_intensity, cmap="Blues")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Final Temperature")
    plt.imshow(final_temperature, cmap="YlGnBu")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
