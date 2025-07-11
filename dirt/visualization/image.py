import numpy as np

import jax.numpy as jnp

def jax_to_image(image):
    return np.array(jnp.clip(
        image, min=0.,  max=1.) * 255).astype(np.uint8)
