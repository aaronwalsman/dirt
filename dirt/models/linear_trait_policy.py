import jax.numpy as jnp

from mechagogue.nn.layer import make_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.static import static_functions
from mechagogue.nn.linear import linear_layer
from mechagogue.nn.initializers import kaiming, zero
from mechagogue.nn.distributions import categorical_sampler_layer

from dirt.envs.tera_arium import TeraAriumTraits
from dirt.models.encoder import make_flatten_encoder

def linear_network(
    in_channels,
    out_channels,
    use_bias=False,
    init_weights=kaiming,
    init_bias=zero,
    dtype=jnp.float32,
):
    
    network = layer_sequence((
        make_flatten_encoder(),
        linear_layer(
            in_channels,
            out_channels,
            use_bias=use_bias,
            init_weights=init_weights,
            init_bias=init_bias,
            dtype=dtype,
        ),
        categorical_sampler_layer(),
    ))
    
    return network

def make_linear_trait_policy(num_actions, dtype=jnp.float32):
    network = linear_network(num_actions, dtype=dtype)
    
    @static_functions
    class LinearTraitPolicy:
        def init(key):
            model_state = network.init(key)
            traits = TeraAriumTraits.default(())
            return model_state, traits
        
        def act(key, obs, state):
            model_state, traits = state
            return network.forward(key, obs, model_state)
        
        def traits(state):
            return state[1]
    
    return LinearTraitPolicy
