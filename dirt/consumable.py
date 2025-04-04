import jax.numpy as jnp

from mechagogue.static_dataclass import static_dataclass

@static_dataclass
class Consumable:
    water : jnp.ndarray
    energy : jnp.ndarray
    biomass : jnp.ndarray
