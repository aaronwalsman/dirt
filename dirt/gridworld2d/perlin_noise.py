import jax.numpy as jnp
import jax.random as jrng
from jax import jit

'''
Simplified Version for Perlin noise using JAX,
where the original package used numpy

'''

def fade(t):
    """
    Smoothstep function for easing.
    """
    return t * t * t * (t * (t * 6 - 15) + 10)

def gradient(h, x, y):
    """
    Calculate gradient based on hashed value.
    """
    vectors = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])  # Gradient vectors
    g = vectors[h % 4]
    return g[..., 0] * x + g[..., 1] * y

@jit
def lerp(t, a, b):
    """
    Linear interpolation.
    """
    return a + t * (b - a)

@jit
def perlin_noise(key, coords):
    """
    Generate Perlin noise for a grid of points.

    key: PRNG key for random gradients.
    coords: A 2D array of shape (N, 2) representing coordinates.

    """
    # Points surrounding each coordinate
    xi = jnp.floor(coords[:, 0]).astype(int)
    yi = jnp.floor(coords[:, 1]).astype(int)

    # Relative positions within the lattice cell
    xf = coords[:, 0] - xi
    yf = coords[:, 1] - yi

    # Smooth the coordinates
    u = fade(xf)
    v = fade(yf)

    # Random gradient hashing
    def hash_fn(x, y):
        """
        Hash function for gradient lookup.
        """
        return (x * 31 + y * 73) % 256

    # Generate random gradients
    key, subkey = jrng.split(key)
    grad_table = jrng.randint(subkey, (256,), minval=0, maxval=256)

    # Hash the corners of the cell
    h00 = grad_table[hash_fn(xi, yi) % 256]
    h10 = grad_table[hash_fn(xi + 1, yi) % 256]
    h01 = grad_table[hash_fn(xi, yi + 1) % 256]
    h11 = grad_table[hash_fn(xi + 1, yi + 1) % 256]

    # Compute dot products at the corners
    g00 = gradient(h00, xf, yf)
    g10 = gradient(h10, xf - 1, yf)
    g01 = gradient(h01, xf, yf - 1)
    g11 = gradient(h11, xf - 1, yf - 1)

    # Interpolate
    nx0 = lerp(u, g00, g10)
    nx1 = lerp(u, g01, g11)
    nxy = lerp(v, nx0, nx1)

    return nxy
