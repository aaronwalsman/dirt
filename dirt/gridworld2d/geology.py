import jax.numpy as jnp
import jax.random as jrng
import chex
from perlin_noise import perlin_noise
from jax import vmap

from typing import Tuple

'''
The total world size is set to be (x,y,h)
On the first pass, I will follow the guide, creating the world
    with fractal noise.

As an add-on, we can adjusted the generated terrain with really high bar 
at the margin to prevent water from flowing out
(it's good even if we don't call the adjusted_margin)

'''

def fractal_noise(
    key : chex.PRNGKey,
    world_size : Tuple[int, int],
    octaves : int,
    lacunarity : float,
    persistence: float,
    grid_unit_scale : float = 0.005,
    height_scale : float = 50,
) -> jnp.ndarray :
    '''
    Function to generate fractal noise with calling Perlin noise
    
    grid_size: Size of the grid (width, height)
    octaves: Number of noise layers
    persistence: Amplitude reduction per octave
    lacunarity: Frequency increase per octave
    key: PRNG key for randomness
    
    '''
    
    #world_aspect = world_size[0] / world_size[1]
    x_scale = grid_unit_scale * world_size[0]
    y_scale = grid_unit_scale * world_size[1]
    x = jnp.linspace(0, x_scale, world_size[0])
    y = jnp.linspace(0, y_scale, world_size[1])
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
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

    return jnp.sum(octave_noises, axis=0).reshape(world_size) * height_scale

def adjusted_margin(
    current_terrain: jnp.ndarray
) -> jnp.ndarray:
    margin = current_terrain.sum()
    current_terrain = current_terrain.at[0, :].set(margin)
    current_terrain = current_terrain.at[-1, :].set(margin)
    current_terrain = current_terrain.at[:, 0].set(margin)
    current_terrain = current_terrain.at[:, -1].set(margin)
    return current_terrain

if __name__ == '__main__':
    key = jrng.PRNGKey(1022)
    noise = fractal_noise(key=key, world_size=(256,256), octaves = 6, persistence = 0.5, lacunarity = 2.0)
    '''
    # noise = adjusted_margin(noise)
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


