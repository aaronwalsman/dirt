from typing import Optional, Tuple, Any

import jax
import jax.numpy as jnp
import jax.random as jrng
import chex
from dirt.gridworld2d.perlin_noise import perlin_noise

from dirt.constants import DEFAULT_FLOAT_DTYPE

'''
The total world size is set to be (x,y,h)
On the first pass, I will follow the guide, creating the world
    with fractal noise.
'''

def fractal_noise(
    key : chex.PRNGKey,
    world_size : Tuple[int, int],
    octaves : int,
    lacunarity : float,
    persistence: float,
    max_octaves : Optional[int] = None,
    grid_unit_scale : float = 0.005,
    height_scale : float = 50,
    dtype : Any = DEFAULT_FLOAT_DTYPE,
) -> jnp.ndarray :
    '''
    Function to generate fractal noise with calling Perlin noise
    
    grid_size: Size of the grid (width, height)
    octaves: Number of noise layers
    persistence: Amplitude reduction per octave
    lacunarity: Frequency increase per octave
    key: PRNG key for randomness
    
    '''
    
    # build the coords
    x_scale = grid_unit_scale * world_size[0]
    y_scale = grid_unit_scale * world_size[1]
    x = jnp.linspace(0, x_scale, world_size[0], dtype=dtype)
    y = jnp.linspace(0, y_scale, world_size[1], dtype=dtype)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    coords = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    
    if max_octaves is None:
        max_octaves = octaves
    keys = jrng.split(key, max_octaves)[:octaves]

    # Frequency and amplitude for each octave
    frequencies = (lacunarity ** jnp.arange(octaves)).astype(dtype)
    amplitudes = (persistence ** jnp.arange(octaves)).astype(dtype)
    
    '''
    # Vectorize across octaves
    def octave_noise(key, frequency, amplitude):
        scaled_coords = coords * frequency
        return amplitude * perlin_noise(key, scaled_coords)
    
    octave_noises = jax.vmap(octave_noise, 
                in_axes=(0, 0, 0))(keys, frequencies, amplitudes)
    
    return jnp.sum(octave_noises, axis=0).reshape(world_size) * height_scale
    '''
    # scan to save memory
    def scan_octave(f, key_frequency_amplitude):
        key, frequency, amplitude = key_frequency_amplitude
        scaled_coords = coords * frequency
        f = f + amplitude * perlin_noise(
            key, scaled_coords)
        return f, None
    
    f, _ = jax.lax.scan(
        scan_octave,
        jnp.zeros((world_size[0]*world_size[1],), dtype=dtype),
        (keys, frequencies, amplitudes),
    )
    
    return f.reshape(world_size) * height_scale
    
