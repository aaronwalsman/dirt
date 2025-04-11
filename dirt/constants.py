import jax.numpy as jnp

DEFAULT_FLOAT_DTYPE = jnp.float32

ROCK_COLOR = jnp.array(
    [0.5, 0.5, 0.5], dtype=DEFAULT_FLOAT_DTYPE)
WATER_COLOR = jnp.array(
    [66/255., 135/255., 245/255.], dtype=DEFAULT_FLOAT_DTYPE)

PHOTOSYNTHESIS_COLOR = jnp.array(
    [66/255., 166/255., 48/255.], dtype=DEFAULT_FLOAT_DTYPE)

DEFAULT_BUG_COLOR = jnp.array(
    [105/255., 57/255., 36/255.], dtype=DEFAULT_FLOAT_DTYPE)

ENERGY_TINT = jnp.array([0.25, 0.25, 0.], dtype=DEFAULT_FLOAT_DTYPE)
BIOMASS_TINT = jnp.array([0, -0.125, -0.25], dtype=DEFAULT_FLOAT_DTYPE)
