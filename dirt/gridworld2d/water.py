from geology import Fractal_Noise
import jax.random as jrng
import jax.numpy as jnp
import jax

from typing import Tuple, Optional, Union

'''
Now that we have got the terrain at hand...
Here is for water!
The tricky part is to deal with the boundary points!(jnp.pad is useful!)
'''

def calculate_flow(
    total_height: jnp.ndarray, 
    water : jnp.ndarray, 
    direction : int,
    flow_rate : float          
) -> jnp.ndarray :
    '''
    Calculate the flow in either of the four directions!(As in the graph)
    directions labeled as: up(0), down(1), left(2), right(3)
    flow rate: fraction of water difference transfer between different areas 
    '''
    padded_height = jnp.pad(total_height, pad_width=1, mode='edge')

    if direction == 0:
        neighbor = padded_height[:-2, 1:-1]
    elif direction == 1:
        neighbor = padded_height[2:, 1:-1]
    elif direction == 2:
        neighbor = padded_height[1:-1, :-2]
    elif direction == 3:
        neighbor = padded_height[1:-1, 2:]
    else:
        raise ValueError("Invalid direction. Must be one of 'up(0)', 'down(1)', 'left(2)', 'right(3)'.")

    diff = total_height - neighbor
    flow = jnp.clip(diff * flow_rate, 0, water)
    return flow

def flow_step(
    terrain: jnp.ndarray, 
    water : jnp.ndarray, 
    flow_rate : float
) -> jnp.ndarray :
    '''
    For each time step, flow the water in all directions by a certain rate
    Rate is input in the calculate function
    '''
    total_height = terrain + water

    flow_up = calculate_flow(total_height, water, 0, flow_rate)
    flow_down = calculate_flow(total_height, water, 1, flow_rate)
    flow_left = calculate_flow(total_height, water, 2, flow_rate)
    flow_right = calculate_flow(total_height, water, 3, flow_rate)

    new_water = (
    water
    - flow_up + jnp.pad(flow_up, ((1, 0), (0, 0)))[1:, :]
    - flow_down + jnp.pad(flow_down, ((0, 1), (0, 0)))[:-1, :]
    - flow_left + jnp.pad(flow_left, ((0, 0), (0, 1)))[:, 1:]
    - flow_right + jnp.pad(flow_right, ((0, 0), (1, 0)))[:, :-1]
    )

    return new_water

def simulate_water_flow(
    terrain: jnp.ndarray, 
    water: jnp.ndarray, 
    time: int, 
    flow_rate: float
) -> jnp.ndarray :
    '''
    Adjust the flow rate of water and time horizon to change how the final terrain looks like
    '''
    water = jax.lax.fori_loop(0, time, lambda i, w: flow_step(terrain, w, flow_rate), water)
    return water

if __name__ == '__main__':
    key = jrng.PRNGKey(1022)
    world_size = (256,256)
    water_initial = 1
    time = 100
    flow_rate = 0.25
    terrain = Fractal_Noise(world_size=world_size, octaves = 6, persistence = 0.5, lacunarity = 2.0, key = key)
    water = jnp.full(world_size, water_initial)
    final_water = simulate_water_flow(terrain, water, time, flow_rate)
    print(final_water)
    import matplotlib.pyplot as plt
    plt.imshow(terrain + final_water, cmap="terrain")
    plt.colorbar(label="Height (Terrain + Water)")
    plt.title("Final Terrain with Water")
    plt.show()
