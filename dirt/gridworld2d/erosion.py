from geology import Fractal_Noise
from water import calculate_flow_twodir, flow_step_twodir
import jax.random as jrng
import jax.numpy as jnp
import jax

from typing import Tuple, Optional, Union

'''
Since we have our basic terrain system and water system...
Why not add a little bit erosion inside the system?
I made it aside from the water to make it easy for add-on trial.(like water flow and rain flow)
Usually we should use erosion with water, but who knows?
'''

def erosion_step_direction(
    terrain: jnp.ndarray, 
    water : jnp.ndarray,
    accumulate_erosion: jnp.ndarray,
    x_offset : int,
    y_offset: int,
    flow_rate: int,
    erosion_endurance: float,
    erosion_ratio: float
) -> jnp.ndarray:
    '''
    Given the current erosion state of each rock,
    we try to decide which rock to erode

    current_erosion: how much amount of water flow through a pixel of terrain
        to a downward location, since only then erosion happens
    erosion_endurance: how much amount of water needed to erode the rock
    erosion_ratio: how much rock are being moved to a certain direction
    '''
    total_height = terrain + water
    current_erosion = calculate_flow_twodir(total_height, water, x_offset, y_offset, flow_rate)
    
    # Aaron's new code to avoid moving rock "uphill"
    padded_terrain = jnp.pad(terrain, pad_width=1, mode='edge')
    neighbor = padded_terrain[
        1 - y_offset : padded_terrain.shape[0] - 1 - y_offset,
        1 + x_offset : padded_terrain.shape[1] - 1 + x_offset,
    ]
    max_erosion = terrain - neighbor
    max_erosion = jnp.where(max_erosion < 0, 0, max_erosion)
    current_erosion = jnp.where(
        current_erosion > max_erosion, max_erosion, current_erosion)
    # end
    
    total_erosion = accumulate_erosion + current_erosion
    erosion_mask = total_erosion > erosion_endurance

    erosion_amount = jnp.where(erosion_mask, current_erosion * erosion_ratio, 0)

    return erosion_amount


def reset_erosion_status(
    terrain: jnp.ndarray,
    previous_terrain: jnp.ndarray,
    current_erosion: jnp.ndarray
) -> jnp.ndarray:
    """
    Reset erosion status for the elements where erosion has occurred.
    """
    erosion_occurred = previous_terrain > terrain
    updated_erosion = jnp.where(erosion_occurred, 0, current_erosion)
    return updated_erosion


def simulate_erosion_step(
    terrain: jnp.ndarray,
    water : jnp.ndarray,
    current_erosion: jnp.ndarray,
    flow_rate: float,
    erosion_endurance: float,
    erosion_ratio: float,
) -> jnp.ndarray:
    '''
    Give the erosion state, erode the rock to all directions
    '''
    offsets = [
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
    ]
    
    erosions = [
        erosion_step_direction(terrain, water, current_erosion, x_offset, y_offset, flow_rate, erosion_endurance, erosion_ratio)
        for x_offset, y_offset in offsets
    ]

    new_terrain = terrain
    for erosion, (x_offset, y_offset) in zip(erosions, offsets):
        new_terrain = new_terrain - erosion
        padded_erosion = jnp.pad(
            erosion,
            (
                (max(0, -y_offset), max(0, y_offset)),
                (max(0, x_offset), max(0, -x_offset)),
            ),
        )
        new_terrain = new_terrain + padded_erosion[
            max(0, y_offset) : erosion.shape[0] + max(0, y_offset),
            max(0, -x_offset) : erosion.shape[1] + max(0, -x_offset),
        ]
    
    return new_terrain


def simulate_erosion(
    terrain: jnp.ndarray,
    water: jnp.ndarray,
    current_erosion: jnp.ndarray,
    flow_rate: int,
    time: int,
    erosion_endurance: float,
    erosion_ratio: float
) -> jnp.ndarray:
    '''
    With the step erosion state and the erosion,
    simulate how erosion will happen with water flow!

    Water Flow-Erosion-Reset Erosion State
    '''
    def erosion_step(carry, _):
        terrain, current_erosion, water = carry

        new_water = flow_step_twodir(terrain, water, flow_rate)
        new_terrain = simulate_erosion_step(terrain, water, current_erosion, flow_rate, erosion_endurance, erosion_ratio)
        current_erosion = reset_erosion_status(new_terrain, terrain, current_erosion)

        return (new_terrain, current_erosion, new_water), None

    initial_state = (terrain, current_erosion, water)
    (final_terrain, final_erosion, final_water), _ = jax.lax.scan(erosion_step, initial_state, None, length=time)

    return final_terrain, final_water

if __name__ == '__main__':
    key = jrng.PRNGKey(1022)
    world_size = (256,256)
    erosion_initial = jnp.zeros(world_size)
    water_initial = 0.5
    time = 200
    flow_rate = 0.25
    terrain = Fractal_Noise(key=key, world_size=world_size, octaves = 6, persistence = 0.5, lacunarity = 2.0)
    water = jnp.full(world_size, water_initial)
    erosion_endurance = 0.2
    erosion_ratio = 0.001

    final_terrain, final_water = simulate_erosion(terrain, water, erosion_initial, flow_rate, time, erosion_ratio, erosion_endurance)

    # print(terrain.sum()) # -349.00833
    # print(final_terrain.sum()) # -349.0083
    # print(terrain == final_terrain) # Erosion is Happening
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.title("Original Terrain")
    plt.imshow(terrain, cmap="terrain")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Eroded Terrain")
    plt.imshow(final_terrain, cmap="terrain")
    plt.colorbar()

    plt.show()
