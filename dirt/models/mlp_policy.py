import jax.numpy as jnp

from mechagogue.nn.layer import make_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.static import static_functions
from mechagogue.nn.initializers import kaiming, zero
from mechagogue.nn.distributions import categorical_sampler_layer
from mechagogue.nn.mlp import mlp


def flatten_bug_observation(obs):
    """
    Flatten a dirt.bug.BugObservation into a single jnp.ndarray.
    
    Expected observation fields and their shapes:
    - age: (batch_size,) -> 1 element per batch
    - newborn: (batch_size,) -> 1 element per batch  
    - rgb: (batch_size, 11, 11, 3) -> 363 elements per batch
    - relative_altitude: (batch_size, 11, 11) -> 121 elements per batch
    - audio: (batch_size, 8) -> 8 elements per batch
    - smell: (batch_size, 8) -> 8 elements per batch
    - wind: (batch_size, 2) -> 2 elements per batch
    - temperature: (batch_size,) -> 1 element per batch
    - external_water: (batch_size,) -> 1 element per batch
    - external_energy: (batch_size,) -> 1 element per batch
    - external_biomass: (batch_size,) -> 1 element per batch
    - health: (batch_size,) -> 1 element per batch
    - internal_water: (batch_size,) -> 1 element per batch
    - internal_energy: (batch_size,) -> 1 element per batch
    - internal_biomass: (batch_size,) -> 1 element per batch
    
    Total flattened dimension: 1 + 1 + 363 + 121 + 8 + 8 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 512
    """
    # Flatten each field and concatenate
    flattened_parts = [
        jnp.atleast_1d(obs.age).flatten(),
        jnp.atleast_1d(obs.newborn).flatten(),
        obs.rgb.flatten(),
        obs.relative_altitude.flatten(),
        obs.audio.flatten(),
        obs.smell.flatten(),
        obs.wind.flatten(),
        jnp.atleast_1d(obs.temperature).flatten(),
        jnp.atleast_1d(obs.external_water).flatten(),
        jnp.atleast_1d(obs.external_energy).flatten(),
        jnp.atleast_1d(obs.external_biomass).flatten(),
        jnp.atleast_1d(obs.health).flatten(),
        jnp.atleast_1d(obs.internal_water).flatten(),
        jnp.atleast_1d(obs.internal_energy).flatten(),
        jnp.atleast_1d(obs.internal_biomass).flatten(),
    ]
    
    # Concatenate all parts
    return jnp.concatenate(flattened_parts)


def mlp_network(
    in_channels,
    out_channels,
    hidden_layers=1,
    hidden_channels=256
):
    """
    This network flattens a BugsObservation, passes it through an MLP,
    then applies a softmax to get a distribution over `out_channels` actions that is then sampled from.
    
    in_channels * hidden_channels + hidden_channels * out_channels
    =
    flattened_bug_observation_dimension * hidden_channels + hidden_channels * num_actions
    =
    512 * 256 + 256 * 14 = 134,656 params
    """
    flatten = make_layer(lambda: None, lambda x: flatten_bug_observation(x))
    network = layer_sequence(
        (
            flatten,
            mlp(
                hidden_layers=hidden_layers,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                use_bias=True,
                p_dropout=0.0,
                init_weights=kaiming,
                init_bias=zero,
                dtype=jnp.float32
            ),
            categorical_sampler_layer()
        )
    )
    
    return network


def make_mlp_policy(obs_dimension, num_actions):
    network = mlp_network(obs_dimension, num_actions)
    
    @static_functions
    class MLPPolicy:
        def init(key):
            return network.init(key)
        
        def act(key, obs, state):
            return network.forward(key, obs, state)
    
    return MLPPolicy
