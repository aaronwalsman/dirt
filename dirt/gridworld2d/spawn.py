from typing import Tuple

import jax.numpy as jnp
import jax.random as jrng

import chex

from dirt.distributions import poisson_vector

def uniform_grid(
    key : chex.PRNGKey,
    world_size : jnp.ndarray,
    p_spawn : float | jnp.ndarray,
) -> jrng.ndarray :
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
    world_size : Tuple[int, int],
    n : int = 1,
) -> jrng.ndarray :
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

def uniform_r(
    key : chex.PRNGKey,
    n : int = 1,
) -> jrng.ndarray :
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
    world_size : Tuple[int, int],
    n : int = 1,
) -> Tuple[jrng.ndarray, jrng.ndarray] :
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
    x = uniform_x(x_key, world_size, n=n)
    key, r_key = jrng.split(key)
    r = uniform_r(r_key, n=n)
    return x, r

def poisson_grid(
    key : chex.PRNGKey,
    world_size : Tuple[int, int],
    mean_n : int,
    max_n : int,
) -> jrng.ndarray :
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
    x = uniform_x(subkey, world_size, max_n)
    
    grid = jnp.zeros(world_size, dtype=jnp.bool)
    grid = grid.at[x[...,0], x[...,1]].set(spawn)
    
    return grid

if __name__ == '__main__':
    key = jrng.key(1234)
    print(poisson_grid(key, (6,6), mean_n=4, max_n=8).astype(jnp.int32))
