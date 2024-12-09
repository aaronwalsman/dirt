import itertools

import jax.numpy as jnp

def bilinear(tensor, axes, *x):
    assert len(axes) == len(x)
    x0 = [jnp.floor(xi).astype(jnp.int32) for xi in x]
    x1 = [xi0 + 1 for xi0 in x0]
    max_size = [tensor.shape[a]-1 for a in axes]
    x0 = [jnp.clip(xi0, 0, mi) for xi0, mi in zip(x0, max_size)]
    x1 = [jnp.clip(xi1, 0, mi) for xi1, mi in zip(x1, max_size)]
    
    
