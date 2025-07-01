import math
from typing import Tuple, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jrng
from jax import vmap

import chex

import dirt.variable_length as vl
import dirt.gridworld2d.dynamics as dynamics

def poisson_vector(
    key : chex.PRNGKey,
    mean_n : int,
    max_n : int,
) -> jnp.ndarray :
    '''
    Samples a number n from a poisson distrbution defined by mean_n (lambda)
    and returns a boolean vector of size max_n where the first n entries are 1
    and the rest are zero.
    
    When used in a jit compiled program, max_n must come from a static variable
    as it controls the shape of a new array.
    
    key : Jax RNG key.
    mean_n (lambda) : The average number of new items to produce.
    max_n : The maximum number of new items to produce.
    '''
    n = jrng.poisson(key, mean_n)
    vector = jnp.arange(max_n) < n

    return vector

def uniform_grid(
    key : chex.PRNGKey,
    p_spawn : Union[float, jnp.ndarray],
    world_size : jnp.ndarray,
) -> jnp.ndarray :
    '''
    Returns a set of spawn locations as boolean entries on a grid, where each
    location is True with probability p_spawn.
    
    When used in a jit compiled program, world_size must come from a static
    variable as it controls the shape of a new array.
    
    key : Jax RNG key.
    world_size : Shape of the resulting grid.
    p_spawn : Probability that any particular element is True.
    '''
    return jrng.uniform(key, shape=world_size, minval=0., maxval=1.) > p_spawn

def uniform_x(
    key : chex.PRNGKey,
    n : int,
    world_size : Tuple[int, int],
) -> jnp.ndarray :
    '''
    Returns a fixed number of spawn positions, where each position is sampled
    uniformly from a grid.
    
    When used in a jit compiled program, n must come from a static variable
    as it controls the shape of a new array.
    
    key : Jax RNG key.
    world_size : Shape to sample positions from.
    n : The number of positions to sample.
    '''
    return jrng.randint(
        key,
        shape=(n,2),
        minval=jnp.array([0,0]),
        maxval=jnp.array(world_size),
    )

def unique_x(
    key : chex.PRNGKey,
    n : int,
    world_size : Tuple[int, int],
    cell_size : Optional[Tuple[int, int]] = None,
) -> jnp.ndarray :
    '''
    '''
    
    # set defaults
    if cell_size is None:
        cell_size = world_size
    
    # checks
    assert world_size[0] % cell_size[0] == 0
    assert world_size[1] % cell_size[1] == 0
    
    # compute the number of cells in each direction (u,v)
    # and the number of elements in each cell
    u = world_size[0] // cell_size[0]
    v = world_size[1] // cell_size[1]
    c = cell_size[0] * cell_size[1]
    
    # for each cell choose (n/(u*v)) internal locations
    items_per_cell = math.ceil(n / (u*v))
    choose = lambda k : jrng.choice(
        k, jnp.arange(c), (items_per_cell,), replace=False)
    key, choice_key = jrng.split(key)
    choice_keys = jrng.split(choice_key, u*v)
    choices = vmap(choose)(choice_keys)
    x = jnp.stack((choices//cell_size[1], choices%cell_size[1]), axis=-1)
    
    # offset the internal locations by the overall location of each cell
    cell_offset = jnp.stack(jnp.meshgrid(
        jnp.arange(0, world_size[0], cell_size[0]),
        jnp.arange(0, world_size[1], cell_size[1]),
        indexing='ij',
    ), axis=-1)
    x = x.reshape(u,v,items_per_cell,2) + cell_offset[:,:,None,:]
    x = x.reshape(-1,2)
    
    # if n was not divisible by the number of cells, we will have generated
    # more locations than we need and must discard locations randomly to
    # get the number we want.
    if x.shape[0] != n:
        key, choice_key = jrng.split(key)
        x = jrng.choice(choice_key, x, (n,), axis=0, replace=False)
    
    return x

def uniform_r(
    key : chex.PRNGKey,
    n : int,
) -> jnp.ndarray :
    '''
    Returns a fixed number of spawn orientations sampled from [0,1,2,3].
    
    When used in a jit compiled program, n must come from a static variable
    as it controls the shape of a new array.
    
    key : Jax RNG key.
    n : The number of orientations to sample.
    '''
    return jrng.randint(
        key,
        shape=(n,),
        minval=0,
        maxval=4,
    )

def unique_xr(
    key : chex.PRNGKey,
    n : int,
    world_size : Tuple[int, int],
    active : Optional[int] = None,
    cell_size : Optional[Tuple[int, int]] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray] :
    '''
    A convenience function that calls both unique_x and uniform_r with the same
    set of parameters.
    
    When used in a jit compiled program, n must come from a static variable
    as it controls the shape of a new array.
    
    key : Jax RNG key.
    world_size : Shape to sample positions from.
    cell_size : Used for unique_x to control uniformity of sampled positions.
    n : The number of positions to sample.
    '''
    key, x_key, r_key = jrng.split(key, 3)
    x = unique_x(x_key, n, world_size, cell_size=cell_size)
    r = uniform_r(r_key, n=n)
    if active is not None:
        x = jnp.where(
            active[:,None], x, jnp.array(world_size, dtype=jnp.int32))
        r = jnp.where(active, r, 0)
    return x, r

def poisson_grid(
    key : chex.PRNGKey,
    mean_n : int,
    max_n : int,
    world_size : Tuple[int, int],
) -> jnp.ndarray :
    '''
    Returns a random number  of spawn locations sampled from a poisson
    distribution as boolean entries on a grid.
    
    When used in a jit compiled program, world_size and max_n must come from
    static variables as they controls the shape of new arrays (max_n does so
    inside the poisson_vector function).
    
    key : Jax RNG key.
    world_size : The shape of the resulting grid.
    mean_n : The poisson lambda parameter indicating the mean number of new
        spawn locations.
    max_n : The maximum number of new spawn locations to add.
    '''
    key, poisson_key = jrng.split(key)
    spawn = poisson_vector(poisson_key, mean_n, max_n)
    
    key, uniform_key = jrng.split(key)
    x = uniform_x(uniform_key, max_n, world_size)
    x = jnp.where(spawn[:,None], x, jnp.array(world_size, dtype=jnp.int32))
    
    grid = jnp.zeros(world_size, dtype=jnp.bool)
    grid = grid.at[x[...,0], x[...,1]].set(spawn)
    
    return grid

def spawn_from_parents(
    reproduce,
    parent_x,
    parent_r,
    world_size=None,
    object_grid=None,
    #child_ids=None,
    child_x_offset=(-1,0),
    child_r_offset=0,
    empty=-1,
):
    '''
    Generate new child positions (child_x) and rotations (child_r) based on
    local offsets from a single parent.  If given a world_size or object_grid
    this will also check to make sure new children are in bounds and do not
    collide with another object or other new children.
    '''
    
    if object_grid is not None:
        assert world_size is None
        world_size = object_grid.shape
    
    child_x, child_r = dynamics.step(
        parent_x,
        parent_r,
        jnp.array(child_x_offset), #[None,:],
        jnp.array(child_r_offset), #[None],
        space='local',
        out_of_bounds='none',
    )
    
    # filter out invalid children
    n, = parent_r.shape
    valid_children = reproduce
    # - first, make sure the child locations would be inside the world
    if world_size is not None:
        inbounds = (
            (child_x[:,0] >= 0) & (child_x[:,0] < world_size[0]) &
            (child_x[:,1] >= 0) & (child_x[:,1] < world_size[1])
        )
        valid_children = valid_children & inbounds
    
    # - second make sure the children will not be on top of any existing objects
    #   or on top of each other
    if object_grid is not None:
        #child_colliders = object_grid[child_x[:,0], child_x[:,1]]
        #valid_children = valid_children & (child_colliders == empty)
        occupancy_map = (object_grid != empty).astype(jnp.int32)
        occupancy_map = occupancy_map.at[child_x[:,0], child_x[:,1]].add(1)
        child_occupancy = occupancy_map[child_x[:,0], child_x[:,1]]
        valid_children = valid_children & (child_occupancy <= 1)
    
    # update the non-reproduced child_x locations to be off the grid so they
    # don't overwrite important values in the object array
    out_of_bounds_x = jnp.array(world_size)[None,:]
    child_x = jnp.where(valid_children[:,None], child_x, out_of_bounds_x)

    #parent_x = parent_x.at[children].set(child_x)
    #parent_r = parent_r.at[children].set(child_r)
    # update the object grid
    '''
    if object_grid is not None:
        assert child_ids is not None
        object_grid = object_grid.at[child_x[...,0], child_x[...,1]].set(
            child_ids)
        return (
            valid_children,
            child_x,
            child_r,
            object_grid,
        )
    '''
    #else:
    return (
        valid_children,
        child_x,
        child_r,
    )

def reproduce_from_parents_old(
    reproduce,
    child_start_id,
    player_id,
    player_parents,
    player_x,
    player_r,
    player_data,
    child_data,
    world_size=None,
    object_grid=None,
    child_x_offset=jnp.array([[-1,0]]),
    child_r_offset=jnp.array([2]),
    empty=-1,
):
    n = player_id.shape[0]
    
    # if an object grid was provided, use the world_size implied by the
    # object grid
    if object_grid is not None:
        world_size = object_grid.shape
    
    # find the new child positions and rotations
    #child_x = player_x + child_x_offset
    #child_r = player_r + child_r_offset
    child_x, child_r = dynamics.step(
        player_x,
        player_r,
        child_x_offset,
        child_r_offset,
        space='local',
        out_of_bounds='none',
    )
    
    # figure out who is able to reproduce
    # first, make sure the reproduce locations would be inside the world
    if world_size is not None:
        inbounds = (
            (child_x[:,0] >= 0) & (child_x[:,0] < world_size[0]) &
            (child_x[:,1] >= 0) & (child_x[:,1] < world_size[1])
        )
        reproduce = reproduce & inbounds
    
    # second make sure the children will not be on top of any existing objects
    if object_grid is not None:
        child_colliders = object_grid[child_x[:,0], child_x[:,1]]
        no_collisions = (child_colliders == empty)
        reproduce = reproduce & (child_colliders == empty)
    
    # update the non-reproduced child_x locations to be off the grid so they
    # don't overwrite important values in the object array
    out_of_bounds_x = jnp.array(world_size)[None,:]
    child_x = jnp.where(reproduce[:,None], child_x, out_of_bounds_x)
    
    # also update the empty player_x locations to be off the grid
    player_x = jnp.where(
        (player_id != empty)[:,None], player_x, out_of_bounds_x)
    
    # concatenate and compact the arrays
    child_id_offsets = jnp.cumsum(reproduce) - 1
    child_id = child_start_id + child_id_offsets
    next_child_start_id = child_id[-1] + 1
    child_id = jnp.where(reproduce, child_id, -1)
    child_parents = player_id
    player_new = jnp.zeros_like(reproduce)
    all_id, (all_parents, all_x, all_r, all_data, all_new) = vl.concatenate(
        (player_id, child_id),
        (
            (player_parents, player_x, player_r, player_data, player_new),
            (child_parents, child_x, child_r, child_data, reproduce),
        ),
    )
    
    # update the object grid
    if object_grid is not None:
        object_grid = object_grid.at[all_x[...,0], all_x[...,1]].set(
            all_id)
        return (
            next_child_start_id,
            all_new,
            all_id,
            all_parents,
            all_x,
            all_r,
            all_data,
            object_grid,
        )
    
    else:
        return (
            next_child_start_id,
            all_new,
            all_id,
            all_parents,
            all_x,
            all_r,
            all_data,
        )

def test_poisson_grid():
    key = jrng.key(1234)
    print(poisson_grid(key, mean_n=4, max_n=8, world_size=(6,6)).astype(jnp.int32))
    print(unique_x(key, 19, (32,32), (8,16)))

def test_reproduce():
    reproduce = jnp.zeros(8, dtype=jnp.bool)
    reproduce = reproduce.at[jnp.array([0,1,3])].set(True)
    child_start_id = 10
    player_id = jnp.array([1,3,4,8,-1,-1,-1,-1])
    parents = jnp.array([0,0,2,3,-1,-1,-1,-1])
    x = jnp.array([
        [1,2],
        [1,0],
        [2,0],
        [1,3],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
    ])
    r = jnp.array([2,3,0,0,0,0,0,0])
    data = jnp.zeros((8,0))
    child_data = jnp.zeros((8,0))
    
    object_grid = jnp.array([
        [-1,-1,-1,-1],
        [ 3,-1, 1, 8],
        [ 4,-1,-1,-1],
        [-1,-1,-1,-1],
    ])
    
    (
        next_child_id,
        all_new,
        all_id,
        all_parents,
        all_x,
        all_r,
        all_data,
        new_object_grid,
    ) = jax.jit(reproduce_from_parents)(
        reproduce,
        child_start_id,
        player_id,
        parents,
        x,
        r,
        data,
        child_data,
        object_grid=object_grid,
    )
    
    breakpoint()

if __name__ == '__main__':
    test_reproduce()
    '''
    a = (jnp.zeros((8,2)), jnp.zeros((8)))
    b = (jnp.ones((8,2)), jnp.ones((8)))
    
    f = lambda a, b : jnp.concatenate((a,b))
    ab = jax.tree.map(f, a, b)
    breakpoint()
    '''
