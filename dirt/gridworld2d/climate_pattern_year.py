from geology import Fractal_Noise
from water import flow_step_twodir
from erosion import simulate_erosion_step, reset_erosion_status
from naive_weather_system import weather_step
import jax.random as jrng
import jax.numpy as jnp
import jax

from typing import Tuple, Optional, Union

'''
Outside of the day!

We won't make a clear cut between the four seasons to match the real life better,
but make season an interpretation for the temperature change, modeled by:

T = Light Strength(LS) + Light Time(LT) + \epsilon

where the LS and LT are determined by the Earth's orbital revolution, and LT is also determined by the rain status of the region
\epsilon is the perturbation to capture some unexplained weather phenomena

One round roughly consists of 360 days
'''

def day_light_length(
    time: int
) -> float:
    pass

def day_light_strength(
    time: int
) -> float:
    pass

def temperature_step(
    day_light_length: float,
    day_light_strength: float
) -> jnp.ndarray:
    pass

def simulate_climate(
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
    erosion_endurance: float,
    day_light_length_initial: float,
    day_light_strength_initial: float
) -> jnp.ndarray:
    pass

if __name__ == '__main__':
    pass
