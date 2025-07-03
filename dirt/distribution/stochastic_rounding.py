import jax.numpy as jnp
import jax.random as jrng

def stochastic_rounding(key, x, round_dtype=jnp.int32):
    x_lo = jnp.floor(x).astype(round_dtype)
    x_hi = jnp.ceil(x).astype(round_dtype)
    r = jrng.bernoulli(key, x-x_lo)
    return jnp.where(r, x_hi, x_lo)
