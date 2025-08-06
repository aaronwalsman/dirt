import jax.numpy as jnp

from mechagogue.nn.layer import make_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.static import static_functions
from mechagogue.nn.initializers import kaiming, zero
from mechagogue.nn.distributions import categorical_sampler_layer
from mechagogue.nn.mlp import mlp

from dirt.envs.tera_arium import TeraAriumTraits


def make_flatten_encoder():
    
    def flatten_bug_observation(obs):
        """
        Flatten a dirt.bug.BugObservation into a single jnp.ndarray.
        
        Expected observation fields and their shapes:
        - age: (batch_size,) -> 1 element per batch
        - newborn: (batch_size,)
        - rgb: (batch_size, 11, 11, 3) -> 363 elements per batch
        - relative_altitude: (batch_size, 11, 11) -> 121 elements per batch
        - audio: (batch_size, 8)
        - smell: (batch_size, 8)
        - wind: (batch_size, 2)
        - temperature: (batch_size,)
        - external_water: (batch_size,)
        - external_energy: (batch_size,)
        - external_biomass: (batch_size,)
        - health: (batch_size,)
        - internal_water: (batch_size,)
        - internal_energy: (batch_size,)
        - internal_biomass: (batch_size,)
        
        Total flattened dimension: 1 + 1 + 363 + 121 + 8 + 8 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 512
        """
        # Flatten each field and concatenate
        flattened_parts = [
            obs.age.flatten(),
            obs.newborn.flatten(),
            obs.rgb.flatten(),
            obs.relative_altitude.flatten(),
            obs.audio.flatten(),
            obs.smell.flatten(),
            obs.wind.flatten(),
            obs.temperature.flatten(),
            obs.external_water.flatten(),
            obs.external_energy.flatten(),
            obs.external_biomass.flatten(),
            obs.health.flatten(),
            obs.internal_water.flatten(),
            obs.internal_energy.flatten(),
            obs.internal_biomass.flatten(),
        ]
        
        # Concatenate all parts
        return jnp.concatenate(flattened_parts)
    
    return make_layer(lambda : None, lambda x : flatten_bug_observation)
