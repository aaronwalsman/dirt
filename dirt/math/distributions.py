import jax.numpy as jnp
import jax.random as jrng

import chex

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
    vector = jnp.arange(max_n) > n
    return vector
