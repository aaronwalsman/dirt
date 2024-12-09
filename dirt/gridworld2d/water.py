from geology import Fractal_Noise
import jax.random as jrng
import jax.numpy as jnp
import jax

from typing import Tuple, Optional, Union

from dirt.gridworld2d.visualization import visualize_height_water

'''
Now that we have got the terrain at hand...
Here is for water!
The tricky part is to deal with the boundary points!(jnp.pad is useful!)

For both boundary and non-boundary terrain, water mass is the same in the test
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

def calculate_flow_twodir(
    total_height: jnp.ndarray, 
    water : jnp.ndarray, 
    x_offset : int,
    y_offset: int,
    flow_rate : float          
) -> jnp.ndarray :
    '''
    Calculate the flow in either of the four directions!(As in the graph)
    we have directions labeled along x-axis and y-axis
    flow rate: fraction of water difference transfer between different areas 
    '''
    padded_height = jnp.pad(total_height, pad_width=1, mode='edge')

    neighbor = padded_height[
        1 - y_offset : padded_height.shape[0] - 1 - y_offset,
        1 + x_offset : padded_height.shape[1] - 1 + x_offset,
    ]

    diff = total_height - neighbor
    flow = jnp.clip(diff * flow_rate, 0, water)
    return flow

def flow_step_twodir(
    terrain: jnp.ndarray, 
    water : jnp.ndarray, 
    flow_rate : float
) -> jnp.ndarray :
    '''
    For each time step, flow the water in all directions by a certain rate
    Rate is input in the calculate function
    '''
    total_height = terrain + water

    offsets = [
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
    ]

    flows = [
        calculate_flow_twodir(total_height, water, x_offset, y_offset, flow_rate)
        for x_offset, y_offset in offsets
    ]

    new_water = water
    for flow, (x_offset, y_offset) in zip(flows, offsets):
        new_water -= flow
        padded_flow = jnp.pad(
            flow,
            (
                (max(0, -y_offset), max(0, y_offset)),
                (max(0, x_offset), max(0, -x_offset)),
            ),
        )
        new_water += padded_flow[
            max(0, -y_offset) : flow.shape[0] + max(0, -y_offset),
            max(0, -x_offset) : flow.shape[1] + max(0, -x_offset),
        ]

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

def simulate_water_flow_twodir(
    terrain: jnp.ndarray, 
    water: jnp.ndarray, 
    time: int, 
    flow_rate: float
) -> jnp.ndarray :
    '''
    Adjust the flow rate of water and time horizon to change how the final terrain looks like
    Two Direction Version
    '''

    water = jax.lax.fori_loop(0, time, lambda i, w: flow_step_twodir(terrain, w, flow_rate), water)
    return water

if __name__ == '__main__':
    key = jrng.PRNGKey(1022)
    world_size = (256,256)
    water_initial = 0.5
    time = 200
    flow_rate = 0.25
    terrain = Fractal_Noise(world_size=world_size, octaves = 6, persistence = 0.5, lacunarity = 2.0, key = key)
    water = jnp.full(world_size, water_initial)
    
    #visualize_height_water(terrain, water)
    
    #'''
    final_water_tdir = simulate_water_flow_twodir(terrain, water, time, flow_rate)
    final_water = simulate_water_flow(terrain, water, time, flow_rate)
    # print(water.sum())
    # print(final_water.sum())
    # print(final_water_tdir.sum())
    # Ensure that total mass of water is the same.
    # With output to be 32768.0, 32768.0, 32768.0 for this seed
    import matplotlib.pyplot as plt
    plt.imshow(final_water, cmap="terrain")
    # plt.imshow(terrain, cmap="terrain")
    plt.colorbar(label="Height (Water)")
    plt.title("Water")
    plt.show()
    #'''
    
    #from dirt.gridworld2d.visualization import make_height_map_mesh, make_obj
    #vt,ft = make_height_map_mesh(terrain)
    #make_obj(vt, ft, file_path='./terrain.obj')
    
    #terrain_water = terrain + final_water
    #vw,fw = make_height_map_mesh(terrain_water)
    #make_obj(vw, fw, file_path='./water.obj')
