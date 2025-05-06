import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.static_dataclass import static_dataclass

from dirt.constants import DEFAULT_FLOAT_DTYPE

def ou_process(
    sigma : jnp.ndarray,
    theta : jnp.ndarray,
    mu : jnp.ndarray,
    dtype : DEFAULT_FLOAT_DTYPE
):
    channels, = mu.shape
    unit_sigma = (2 * theta * sigma**2)**0.5
    
    def init(
        key : chex.PRNGKey,
    ):
        std = sigma / jnp.sqrt(2 * theta)
        x = mu + jrng.normal(key, (channels,), dtype=dtype) * std #sigma
        return x
    
    def step(
        key : chex.PRNGKey,
        x : jnp.array,
        step_size : float = 1.
    ):
        step_theta = theta * step_size
        step_sigma = unit_sigma * (step_size**0.5)
        
        step_reversion = step_theta * (mu - x)
        step_noise = jrng.normal(key, (channels,), dtype=dtype) * step_sigma
        x = x + step_reversion + step_noise
        
        return x
    
    return init, step

if __name__ == '__main__':
    init_process, step_process = ou_process(
        sigma = jnp.array((3.)),
        theta = jnp.array((0.001)),
        mu = jnp.array((0,)),
    )
    
    def simulate_steps(key, step_size, N):
        key, init_key = jrng.split(key)
        x = init_process(init_key)
        
        def body(key_x, _):
            key, x = key_x
            key, step_key = jrng.split(key)
            x = step_process(step_key, x, step_size)
            return (key, x), x
        
        x, xi = jax.lax.scan(body, (key, x), None, length=N)
        
        return xi
    
    xi = simulate_steps(jrng.key(1234), 0.01, 1000000)
    stdi = jnp.std(xi, axis=0)
    print(stdi)
    print(xi[:8])
        
    xi_int = xi[0::2] + xi[1::2]
    stdi_int = jnp.std(xi_int, axis=0)
    print(stdi_int)
    
    xj = simulate_steps(jrng.key(1236), 0.02,  500000) 
    stdj = jnp.std(xj, axis=0)
    print(stdj)
    
    print(jnp.sum(xi, axis=0))
