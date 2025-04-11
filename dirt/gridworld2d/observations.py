from typing import Tuple

import jax.numpy as jnp

from dirt.gridworld2d.dynamics import rotate

def local_view(
    x : jnp.ndarray,
    r : jnp.ndarray,
    observation_grid : jnp.ndarray,
    view_shape : Tuple[Tuple[int, int], Tuple[int, int]],
    subsample : int = 1,
    out_of_bounds : int | float | jnp.ndarray = 0,
) -> jnp.ndarray :
    '''
    Extracts a local view window around a location x with orientation r.  The
    size of the window is specified as a bounding box containing two corners.
    x : the position of each view window
    r : the orientation of each view window
    observation_grid : the larger grid from which the views should be extracted
    view_shape : ((min_0, min_1), (max_0, max_1)) two corners of a bounding box
        describing the region to be extracted
    subsample : downscale the sampled data
    out_of_bounds : the value that should be inserted in any view that falls
        outside the viewing area
    '''
    # construct the local rows and columns
    (min_0, min_1), (max_0, max_1) = view_shape
    step_size_0 = [subsample,-subsample][max_0-min_0 < 0]
    step_size_1 = [subsample,-subsample][max_1-min_1 < 0]
    rc = jnp.stack(jnp.meshgrid(
        jnp.arange(min_0, max_0, step_size_0),
        jnp.arange(min_1, max_1, step_size_1),
        indexing='ij',
    ), axis=-1)
    
    # construct the global rows and columns by rotating and offsetting the
    # local rows and columns using the agents' positions and orientations
    global_rc = rotate(rc, r[...,None, None]) + x[...,None,None,:]
    
    # convert negative values in global_rc to out-of-bound indices
    too_big = max(observation_grid.shape)
    global_rc = jnp.where(global_rc >= 0, global_rc, too_big)
    
    # cut out regions of the observation grid corresponding to the computed
    # rows and columns
    fpv = observation_grid.at[global_rc[...,0], global_rc[...,1]].get(
        mode='fill', fill_value=out_of_bounds)
    
    # return
    return fpv

def first_person_view(
    x : jnp.array,
    r : jnp.array,
    observation_grid : jnp.array,
    view_width : int,
    view_distance : int,
    view_back_distance : int = 0,
    subsample : int = 1,
    out_of_bounds : int | float | jnp.ndarray = 0,
) -> jnp.ndarray :
    '''
    A reparameterized version of local_view with a different description of the
    local shape of each view to be extracted.
    x : the position of each view window
    r : the orientation of each view window
    observation_grid : the larger grid from which the views should be extracted
    view_width : the width of the view window
    view_distance : the distance in front of the position and orientation to
        extend the view window
    view_back_distance : the distance behind the position and orientation to
        extend the view window
    out_of_bounds : the value that should be inserted in any view that falls
        outside the viewing area
    '''
    w = (view_width-1)//2
    view_shape = ((-view_back_distance,-w), (view_distance,w+1))
    return local_view(
        x,
        r,
        observation_grid,
        view_shape,
        subsample=subsample,
        out_of_bounds=out_of_bounds,
    )

if __name__ == '__main__':
    observation_grid = jnp.arange(5*5).reshape(5,5)
    x = jnp.array([
        [2,2],
        [2,2],
        [2,2],
        [2,2],
    ])
    r = jnp.array([
        0,
        1,
        2,
        3,
    ])
    a = jnp.array([-2,-2])
    b = jnp.array([3,3])
    obs = first_person_view(x, r, observation_grid, 5, 5)
    
    directions = rotate(jnp.array([[1,0]]), jnp.array([0,1,2,3]))
    
    print(observation_grid)
    print('')
    
    for i in range(4):
        print(obs[i])
        print(directions[i])
        print('')
