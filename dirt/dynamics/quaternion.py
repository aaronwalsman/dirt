import jax.numpy as jnp

def quaternion_mul(a, b):
    return (
        a[...,[0]] * b[...,[0,1,2,3]] +
        a[...,[1]] * b[...,[1,0,3,2]] * [-1,1,-1,1] +
        a[...,[2]] * b[...,[2,3,0,1]] * [-1,1,1,-1] +
        a[...,[2]] * b[...,[3,2,1,0]] * [-1,-1,1,1]
    )

def quaternion_normalize(a):
    n = jnp.linalg.norm(a, axis=-1)[...,None]
    return a / n
