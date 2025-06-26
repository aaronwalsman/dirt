import jax.random as jrng
import jax.numpy as jnp
import jax.nn as jnn

from mechagogue.static_dataclass import static_dataclass
from mechagogue.nn.linear import (
    embedding_layer, linear_layer, conv_layer)
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.mlp import mlp
from mechagogue.nn.structured import parallel_dict_layer
from mechagogue.nn.distributions import categorical_sampler_layer
from mechagogue.nn.debug import print_activations_layer

from dirt.envs.nomnom import NomNomObservation, NomNomAction

'''
We want low, but non-zero epistasis.  What does a normal linear layer have:

W00 W01 W02 W03   X0   X'0
W10 W11 W12 W13 * X1 = X'1
W20 W21 W22 W23   X2   X'2
W30 W31 W32 W33   X3   X'3

So W00-W03 interact with each other and X0-X3 to produce X'0.

If you allowed no interaction, you would only have a 4-vector weight matrix.

You could also have a block weight matrix where only two could interact:

W00 W01   0   0   X0   X'0
W10 W11   0   0 * X1 = X'1
  0   0 W22 W23   X2   X'2
  0   0 W32 W33   X3   X'3

So here only the first two interact to produce X'0 and X'1 and the second two
interact to produce X'2 and X'3.  If you keep doing this forever though, it's
like you are running to separate neural networks.  So what you could do is
follow this up with:

W00   0   0 W03   X0   X'0
  0 W11 W12   0 * X1 = X'1
  0 W21 W22   0   X2   X'2
W30   0   0 W33   X3   X'3

Basically change around the block structure at each layer to get different
things to interact with each other, but mix more slowly.

You could also do this by fixing the block structure and permuting X.  This is
the same idea used in shufflenet.

--------------------------------

What if we come back to the idea of summing a bunch of random matrices?
Possibly with weights.  By controlling the sparsity of those matrices, we could
get a kind of epistasis, right?  Oooooh...

So how do you do it?  What if each matrix was some weights and a randomized
sparsity pattern.  Each weight entry would be a sum of a bunch of zeros and a
few normally distributed numbers.  Could you show that each weight would be
normally distributed?  A sum of normals is a normal, but what about a sum of
normals with a random number of them set to zero.  Or better yet, is there a
different distribution I could use to draw the weights such that it would be
normally distributed?  Why do I care if it's normally distributed again?
'''

# utilities
relu = (lambda: None, jnn.relu)

def nomnom_action_decoder(in_channels, dtype=jnp.float32):
    return layer_sequence((
        (
            lambda: None,
            lambda x : {'forward' : x, 'rotate' : x, 'reproduce' : x}
        ),
        parallel_dict_layer({
            'forward' : layer_sequence((
                linear_layer(in_channels, 2, use_bias=True, dtype=dtype),
                categorical_sampler_layer(),
            )),
            'rotate' : layer_sequence((
                linear_layer(in_channels, 3, use_bias=True, dtype=dtype),
                categorical_sampler_layer(choices=jnp.array([-1,0,1])),
            )),
            'reproduce' : layer_sequence((
                linear_layer(in_channels, 2, use_bias=True, dtype=dtype),
                categorical_sampler_layer(),
            )),
        }),
        (lambda: None, lambda x : NomNomAction(**x)),
    ))

#def penumbra_layer(
#    #initial_value,
#    max_things,
#    width,
#):
#    #def init():
#    #    return initial_value
#    
#    def model(state):
#        x = jnp.arange(max_things, dtype=state.dtype) - state - width + 1
#        x = -x / width
#        x = jnp.clip(x, min=0., max=1.)
#        return x
#    
#    return init, model

def soft_mask(x, n, width):
    mask = -(jnp.arange(n, dtype=x.dtype) - x - width + 1) / width
    mask = jnp.clip(mask, min=0., max=1.)
    return mask

def soft_masked_linear_layer(
    initial_in_channels,
    max_in_channels,
    in_soft_width,
    initial_out_channels,
    max_out_channels,
    out_soft_width,
    dtype=jnp.float32,
):
    init_linear, linear = linear_layer(
        max_in_channels, max_out_channels, dtype=dtype)
    
    def init(key):
        linear_state = init_linear(key)
        return (initial_in_channels, initial_out_channels, linear_state)
    
    def model(x, state):
        in_channels, out_channels, linear_state = state
        
        in_mask = soft_mask(in_channels, max_channels, in_soft_width)
        x = x * in_sigmoid
        
        x = linear(x, linear_state)
        
        out_mask = soft_mask(out_channels, max_channels, out_soft_width)
        x = x * out_sigmoid
        
        return x
    
    return init, model

def soft_masked_resnet(
    initial_channels,
    max_channels,
    soft_channel_width,
    initial_expansion,
    max_expansion,
    soft_expansion_width,
    initial_depth,
    max_depth,
    soft_depth_width,
):
    def residual_block():
        expand_init, expand = soft_masked_linear_layer(
            initial_in_channels,
            max_in_channels,
            soft_channel_width,
            initial_expansion,
            max_expansion,
            soft_expansion_width,
        )
        
        contract_init, contract = soft_masked_linear_layer(
            initial_expansion,
            max_expansion,
            soft_expansion_width,
            initial_in_channels,
            max_in_channels,
            soft_channel_width,
        )
            
    blocks = [residual_block() for _ in max_depth]
    
    def model(x, state):
        depth, block_states = state
        depth_mask = soft_mask(depth, max_depth, soft_depth_width)
        
        for (_, model_block), block_state, mask in zip(blocks, block_states, depth_mask):
            x1 = model_block(x, block_state)
            x = x + x1 * mask

# params
@static_dataclass
class NomNomModelParams:
    num_input_classes : int = 4
    view_width : int = 5
    view_distance : int = 5

# models
def nomnom_unconditional_model(params=NomNomModelParams(), dtype=jnp.float32):
    return layer_sequence((
        (lambda : None, lambda x : jnp.ones_like(x.energy)),
        nomnom_action_decoder(1, dtype=dtype),
    ))

def nomnom_linear_model(params=NomNomModelParams(), dtype=jnp.float32):
    in_dim = (
        (params.num_input_classes-1) *
        params.view_width *
        params.view_distance +
        1   # energy
    )
    out_dim = 2+3+2
    
    def encoder_forward(x):
        food = (x.view == 1).reshape(-1).astype(dtype)
        players = (x.view == 2).reshape(-1).astype(dtype)
        out_of_bounds = (x.view == 3).reshape(-1).astype(dtype)
        return jnp.concatenate(
            (food, players, out_of_bounds, x.energy[...,None]), axis=-1)
    
    encoder = (lambda: None, encoder_forward)
    decoder = nomnom_action_decoder(in_dim, dtype=dtype)
    
    return layer_sequence((
        encoder,
        decoder,
    ))

def nomnom_model(parms=NomNomModelParams()):
    
    # encoder
    # - view encoder
    assert params.view_distance % params.view_patch_size == 0
    assert params.view_width % params.view_patch_size == 0
    distance_patches = params.view_patch_size // view.patch_size
    width_patches = params.view_width // view.patch_size
    view_patch_embedding = conv_layer(
        in_channels,
        params.view_hidden_channels,
        kernel_size=params.view_patch_size,
        stride=params.view_patch_size,
        use_bas=params.view_patch_bias,
        dtype=params.dtype,
    )
    view_linear = linear_layer(...)

def nomnom_model(params=NomNomModelParams()):
    
    # encoder
    #   this section builds the components that convert the inputs to a fixed
    #   hidden dimension that can be processed by the backbone
    # - view encoder
    view_embedding = embedding_layer(params.num_input_classes, 32)
    view_conv1 = conv_layer(32, 64, padding='VALID')
    view_conv2 = conv_layer(64, 128, padding='VALID')
    flattened_channels = (params.view_width-4) * (params.view_distance-4) * 128
    flatten = (lambda: None, lambda x : x.reshape(flattened_channels))
    view_fc1 = linear_layer(flattened_channels, 128)
    view_encoder = layer_sequence(
        (view_embedding, view_conv1, relu, view_conv2, flatten, relu, view_fc1)
    )
    
    # - energy encoder
    energy_encoder = layer_sequence((
        (lambda: None, lambda x : x.reshape(-1)),
        mlp(
            hidden_layers=1,
            in_channels=1,
            hidden_channels=32,
            out_channels=128,
        ),
    ))
    
    # - combine the encoders
    #   this first converts an observation object to a dictionary
    #   then passes that dictionary to the corresponding encoders
    #   then sums the output
    encoder = layer_sequence((
        (lambda: None, lambda x : {'view': x.view, 'energy': x.energy}),
        parallel_dict_layer({'view':view_encoder, 'energy':energy_encoder}),
        (lambda: None, lambda x : sum(x.values())),
        relu,
    ))
    
    # backbone
    #   this is a small MLP
    backbone = layer_sequence((
        linear_layer(128, 64),
        relu,
        linear_layer(64, 32),
        relu,
        (lambda: None, lambda x : {'forward':x, 'rotate':x, 'reproduce':x}),
    ))
    
    # decoder heads
    #   the decoders convert the output of the MLP to the individual
    #   components of the action
    # - these are three linear layers followed by categorical samplers
    decoder_heads = parallel_dict_layer({
        'forward' : layer_sequence((
            linear_layer(32, 2),
            #print_activations_layer('forward:'),
            categorical_sampler_layer(),
            #print_activations_layer('forward_sample:'),
        )),
        'rotate' : layer_sequence((
            linear_layer(32, 3),
            #print_activations_layer('rotate:'),
            categorical_sampler_layer(choices=jnp.array([-1,0,1])),
            #print_activations_layer('rotate_sample:'),
        )),
        'reproduce' : layer_sequence((
            linear_layer(32, 2),
            #print_activations_layer('reproduce:'),
            categorical_sampler_layer(),
            #print_activations_layer('reproduce_sample:'),
        )),
    })
    
    # - this runs the three decoders and then combines the result into a
    #   new action
    decoder = layer_sequence((
        decoder_heads,
        (lambda: None, lambda x : NomNomAction(**x)),
        #print_activations_layer('action:'),
    ))
    
    # combine the encoder, backbone and decoder
    return layer_sequence([
        encoder,
        backbone,
        decoder,
    ])

def test_model(key):
    init_model, model = nomnom_model()
    
    model_state = init_model(key)
    observation = NomNomObservation(
        view=jnp.zeros((5,5), dtype=jnp.int32),
        energy=jnp.array([1.]),
    )
    action = model(key, observation, model_state)
    print(action)

if __name__ == '__main__':
    test_model(jrng.key(12345))
