import jax.numpy as jnp

from mechagogue.static import static_data

@static_data
class Consumable:
    water : jnp.ndarray
    energy : jnp.ndarray
    biomass : jnp.ndarray
