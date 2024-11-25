import jax.numpy as jnp
import jax.random as jrng
import chex
from perlin_noise import perlin_noise
from jax import vmap

from typing import Tuple, Optional, Union

# Used for random generating(Come in Handy)
# key = jrng.key(1022)
# key1, key2 = jrng.split(key)
# test = jrng.normal(key, (100,100,))
# print(test[0:5])

'''
The total world size is set to be (x,y,h)
On the first pass, I will follow the guide, creating the world
    with fractal noise.

'''

def Fractal_Noise(
    world_size : Tuple[int, int],
    octaves : int,
    lacunarity : float,
    persistence: float,
    key : chex.PRNGKey,
) -> jnp.ndarray :
    '''
    Function to generate fractal noise with calling Perlin noise
    
    grid_size: Size of the grid (width, height)
    octaves: Number of noise layers
    persistence: Amplitude reduction per octave
    lacunarity: Frequency increase per octave
    key: PRNG key for randomness
    
    '''
    x = jnp.linspace(0, 10, world_size[0])
    y = jnp.linspace(0, 10, world_size[1])
    xx, yy = jnp.meshgrid(x, y)
    coords = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    keys = jrng.split(key, num=octaves)

    # Frequency and amplitude for each octave
    frequencies = lacunarity ** jnp.arange(octaves)
    amplitudes = persistence ** jnp.arange(octaves)

    # Vectorize across octaves
    def octave_noise(key, frequency, amplitude):
        scaled_coords = coords * frequency
        return amplitude * perlin_noise(key, scaled_coords)
    
    octave_noises = vmap(octave_noise, 
                in_axes=(0, 0, 0))(keys, frequencies, amplitudes)

    return jnp.sum(octave_noises, axis=0).reshape(world_size)

if __name__ == '__main__':
    key = jrng.PRNGKey(1234)
    noise = Fractal_Noise(world_size=(256,256), octaves = 6, persistence = 0.5, lacunarity = 2.0, key = key)
    '''
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(noise, cmap="terrain", extent=(0, 10, 0, 10))
    plt.colorbar(label="Height")
    plt.title("Fractal Noise with JAX")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    '''
    
    from dirt.gridworld2d.visualization import make_height_map_mesh, make_obj
    vertices, faces = make_height_map_mesh(noise*10)
    make_obj(vertices, faces, './tmp.obj')


