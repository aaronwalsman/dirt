from geology import fractal_noise
from water import flow_step_twodir
from erosion import simulate_erosion_step, reset_erosion_status
from naive_weather_system import weather_step
import jax.random as jrng
import jax.numpy as jnp
import jax

from typing import Tuple, Optional, Union

'''
Then it's time for the climate system to work!
In expectation, Half time will have daylight while the other half...
Let's give some ambient lighting, likely from the moon!

And this is going to be shifted by the season parameter, since in summer we have longer days
The season shift will be accomplished in the climate_patter_year.py file

One day is accomplished of 24 steps, matching the 24 hrs
'''

def get_day_status(
    day_light_length: int,
    time: int
) -> int:
    '''
    Based on the length of day light, determine the status of the day

    day_light_length: [0,24]
    
    1: Sun
    0: Moon
    '''
    edge = time % 24
    return jnp.where(edge <= day_light_length, 1, 0)

def get_angle(
    day_light_length: int,
    time: int
) -> float:
    '''
    Based on the status of the day and the time now, determine the angle of the light

    angle is the one with positive right axis
    '''
    time %= 24
    angle = jnp.where(
        time <= 12,
        (time / day_light_length) * jnp.pi,
        ((time - day_light_length) / (24 - day_light_length)) * jnp.pi
    )
    return angle
    
def terrain_gradient(
    terrain: jnp.ndarray
) -> jnp.ndarray:
    '''
    Get the terrain normals from the gradients
    '''
    dx, dy = jnp.gradient(terrain)
    normals = jnp.stack((-dx, -dy, jnp.ones_like(terrain)), axis=-1)

    magnitude = jnp.linalg.norm(normals, axis=-1, keepdims=True)
    magnitude = jnp.where(magnitude == 0, 1e-8, magnitude)

    return normals / magnitude

def get_normalize(
    array: jnp.ndarray
) -> jnp.ndarray:
    min_val = jnp.min(array)
    max_val = jnp.max(array)
    return (array - min_val) / (max_val - min_val + 1e-8)

def light_step(
    terrain: jnp.ndarray,
    water: jnp.ndarray,
    light_strength: float,
    day_light_length: int,
    day_status: int,
    time: int,
    night_effect = 0.1
) -> jnp.ndarray:
    '''
    light_strength: determined by the distance between the terrain and the Sun

    returns the light of every pixel

    Get the Idea from website: https://learnopengl.com/Lighting/Basic-Lighting
    '''
    angle = get_angle(day_light_length, time)
    light_direction = jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])
    final_terrain = terrain + water
    normals = terrain_gradient(final_terrain)
    light_direction = light_direction.reshape(1, 1, 3)
    dot_products = jnp.einsum('ijk,ijk->ij', normals, light_direction)
    dot_products_norm = get_normalize(dot_products)
    light_intensity = jnp.clip(dot_products_norm * light_strength, 0, 1)
    return jnp.where(time % 24 <= 12, light_intensity, night_effect * light_intensity)


def absorb_temp(
    light_intensity: jnp.ndarray,
    water: jnp.ndarray,
    temperature: jnp.ndarray,
    rain_status: jnp.ndarray,
    current_evaporation: jnp.ndarray,
    water_effect = 0.5,
    rain_effect = 0.1, 
    evaporation_effect = 0.5
) -> jnp.ndarray:
    '''
    Absorb temperature in the daylight

    Here we assume that the difference in terrain height won't affect the light intensity so much

    water_effetct: larger value means weaker effects, [0, \infty]
    rain_effect: larger value means weaker effects, [0, 1]
    evaporation_effect : larger value means weaker effects, [0, \infty]
    '''
    water_norm = get_normalize(water)
    current_evaporation_norm = get_normalize(current_evaporation)
    absorption_rate = light_intensity * (1 - (1 - rain_effect) * rain_status) * (1 + water_effect - water_norm) * (1 + evaporation_effect - current_evaporation_norm)
    return temperature + absorption_rate

def release_temp(
    light_intensity: jnp.ndarray,
    water: jnp.ndarray,
    temperature: jnp.ndarray,
    rain_status: jnp.ndarray,
    current_evaporation: jnp.ndarray,
    night_effect,
    water_effect,
    rain_effect, 
    evaporation_effect
) -> jnp.ndarray:
    '''
    Release temperature in the night
    '''
    water_norm = get_normalize(water)
    current_evaporation_norm = get_normalize(current_evaporation)
    release_rate = light_intensity/night_effect * (1 - (1 - rain_effect) * rain_status) * (1 + water_effect - water_norm) * (1 + evaporation_effect - current_evaporation_norm)
    return temperature - release_rate

def temperature_step(
    time: int,
    water: jnp.ndarray,
    temperature: jnp.ndarray,
    rain_status: jnp.ndarray,
    light_intensity: jnp.ndarray,
    current_evaporation: jnp.ndarray,
    day_status: int, 
    night_effect: float, 
    water_effect: float, 
    rain_effect: float, 
    evaporation_effect: float
) -> jnp.ndarray:
    '''
    Simulate the per step light and temperature system
    '''
    return jnp.where(time % 24 <= 12, absorb_temp(light_intensity, water, temperature, rain_status, current_evaporation, water_effect, rain_effect, evaporation_effect), release_temp(light_intensity, water, temperature, rain_status, current_evaporation, night_effect, water_effect, rain_effect, evaporation_effect)
)

def simulate_full_weather_day(
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
    light_strength: jnp.ndarray,
    light_length: int,
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
        new_day_status = get_day_status(light_length, current_time)
        new_light_intensity = light_step(terrain, water, light_strength, light_length, new_day_status, current_time) #Porblem of getting None

        # 4. Temperature
        new_temperature = temperature_step(current_time, water, current_temperature, rain_status, light_intensity, current_evaporation, day_status, night_effect, water_effect, rain_effect, evaporation_effect)
        
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
    terrain = fractal_noise(world_size=world_size, octaves = 6, persistence = 0.5, lacunarity = 2.0, key = key)
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
    night_effect = 0.17 # Under this, the temperature system doesn't seem to explode
    water_effect = 0.5
    rain_effect = 0.1 
    evaporation_effect = 0.5
    light_length = 12
    light_strength = 1

    final_terrain, final_water, left_evaporation, final_rain_status, final_erosion, final_day_status, final_light_intensity, final_temperature = simulate_full_weather_day(
        terrain, water, temperature_initial, time, 
        erosion_initial, rain_initial, evaporation_initial, day_status_initial, light_intensity_initial,  
        evaporate_rate, air_up_limit, air_down_limit, rain,
        flow_rate, erosion_ratio, erosion_endurance, light_strength, light_length, night_effect, water_effect, rain_effect, evaporation_effect
    )
    total_water = final_water + left_evaporation
    # print(terrain.sum()) # -349.00833
    # print(final_terrain.sum()) # -349.00824
    # print(water.sum()) # 32768.0
    # print(total_water.sum()) # 32767.992
    # print(final_terrain == terrain) # Erosion happening with this paramter

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



