import jax.numpy as jnp

from mechagogue.nn.layer import make_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.static import static_functions
from mechagogue.nn.linear import linear_layer
from mechagogue.nn.initializers import kaiming, zero
from mechagogue.nn.distributions import categorical_sampler_layer


def blind_network(
    out_channels,
    use_bias=False,
    init_weights=kaiming,
    init_bias=zero,
    dtype=jnp.float32
):
    """
    This network maps any input to 1, applies a 1 x out_channels linear layer,
    then applies a softmax to get a probability distribution over the out_channels.
    """
    # map_to_one layer's `init` does nothing
    # its `forward` returns a jnp vector of one 1
    map_to_one = make_layer(lambda : None, lambda x : jnp.ones((1,)))
    
    in_channels = 1
    network = layer_sequence((
            map_to_one,
            linear_layer(
                in_channels,
                out_channels,
                use_bias=use_bias,
                init_weights=init_weights,
                init_bias=init_bias,
                dtype=dtype,
            ),
            categorical_sampler_layer(),  # softmax over `out_channels` logits
        ))
    
    return network


def make_blind_policy(num_actions):
    network = blind_network(num_actions)
    
    @static_functions
    class BlindPolicy:
        def init(key):
            return network.init(key)
        
        def act(key, obs, state):
            # the network is blind to the observation - it always maps it to 1,
            # then applies 1 x num_actions linear layer,
            # then applies a softmax to get a probability distribution over the num_actions logits,
            # then samples from the distribution to get an action
            return network.forward(key, obs, state)
    
    return BlindPolicy
