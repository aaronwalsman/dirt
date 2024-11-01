import math
from typing import Tuple, Optional

import jax.numpy as jnp
import jax.random as jrng
from jax import vmap

import chex

from dirt.math.distributions import poisson_vector

def uniform_grid(
    key : chex.PRNGKey,
    p_spawn : float | jnp.ndarray,
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

def uniform_xr(
    key : chex.PRNGKey,
    n : int,
    world_size : Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray] :
    '''
    A convenience function that calls both uniform_x and uniform_r with the same
    set of parameters.
    
    When used in a jit compiled program, n must come from a static variable
    as it controls the shape of a new array.
    
    key : Jax RNG key.
    world_size : Shape to sample positions from.
    n : The number of positions to sample.
    '''
    key, x_key = jrng.split(key)
    x = uniform_x(x_key, n, world_size)
    key, r_key = jrng.split(key)
    r = uniform_r(r_key, n)
    return x, r

def unique_xr(
    key : chex.PRNGKey,
    n : int,
    world_size : Tuple[int, int],
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
    key, x_key = jrng.split(key)
    x = unique_x(x_key, n, world_size, cell_size=cell_size)
    key, r_key = jrng.split(key)
    r = uniform_r(r_key, n=n)
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
    key, subkey = jrng.split(key)
    spawn = poisson_vector(subkey, mean_n, max_n)
    
    key, subkey = jrng.split(key)
    x = uniform_x(subkey, max_n, world_size)
    
    grid = jnp.zeros(world_size, dtype=jnp.bool)
    grid = grid.at[x[...,0], x[...,1]].set(spawn)
    
    return grid

if __name__ == '__main__':
    key = jrng.key(1234)
    print(poisson_grid(key, mean_n=4, max_n=8, world_size=(6,6)).astype(jnp.int32))
    print(unique_x(key, 19, (32,32), (8,16)))
