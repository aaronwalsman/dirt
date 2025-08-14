import jax.numpy as jnp

from mechagogue.nn.layer import make_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.static import static_functions
from mechagogue.nn.initializers import kaiming, zero
from mechagogue.nn.distributions import categorical_sampler_layer
from mechagogue.nn.mlp import mlp

from fitness_of_intelligence.models.mutable_mlp import make_mutable_mlp

from dirt.envs.tera_arium import TeraAriumTraits

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


def make_mutable_mlp_trait_policy(
    in_channels,
    out_channels,
    initial_hidden_channels,
    min_hidden_channels,
    max_hidden_channels,
    initial_hidden_layers,
    max_hidden_layers,
    use_bias,
    weight_mutation_rate=1e-3,
    bias_mutation_rate=1e-3,
    channel_mutation_rate=0.05,
    layer_mutation_rate=0.01,
):
    mutable_mlp = make_mutable_mlp(
        in_channels,
        out_channels,
        initial_hidden_channels,
        min_hidden_channels,
        max_hidden_channels,
        initial_hidden_layers,
        max_hidden_layers,
        use_bias,
        weight_mutation_rate,
        bias_mutation_rate,
        channel_mutation_rate,
        layer_mutation_rate,
    )
    
    flatten = make_layer(lambda: None, lambda x: flatten_bug_observation(x))
    network = layer_sequence(
        (
            flatten,
            mutable_mlp,
            categorical_sampler_layer(),
        )
    )
    
    @static_functions
    class MLPTraitPolicy:
        def init(key):
            model_state = network.init(key)
            traits = TeraAriumTraits.default(())
            return model_state, traits
        
        def act(key, obs, state):
            model_state, traits = state
            return network.forward(key, obs, model_state)
        
        def traits(state):
            return state[1]
    
    return MLPTraitPolicy
