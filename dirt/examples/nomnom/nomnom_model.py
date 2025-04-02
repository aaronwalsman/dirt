import jax.random as jrng
import jax.numpy as jnp
import jax.nn as jnn
import jax.tree_map as tree_map

from mechagogue.static_dataclass import static_dataclass
from mechagogue.nn.linear import (
    embedding_layer, linear_layer, conv_layer)
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.mlp import mlp
from mechagogue.nn.structured import parallel_dict_layer
from mechagogue.nn.distributions import categorical_sampler_layer
from mechagogue.nn.debug import print_activations_layer

from dirt.envs.nomnom import NomNomObservation, NomNomAction

@static_dataclass
class NomNomModelParams:
    num_input_classes : int = 4
    view_width : int = 5
    view_distance : int = 5

def nomnom_linear_model(params=NomNomModelParams()):
    in_dim = (
        (params.num_input_classes-1) *
        params.view_width *
        params.view_distance +
        1   # energy
    )
    out_dim = 2+3+2
    
    def encoder_forward(x):
        food = (x.view == 1).reshape(-1).astype(jnp.float32)
        players = (x.view == 2).reshape(-1).astype(jnp.float32)
        out_of_bounds = (x.view == 3).reshape(-1).astype(jnp.float32)
        x = jnp.concatenate(
            (food, players, out_of_bounds, x.energy[...,None]), axis=-1)
        return {'forward' : x, 'rotate' : x, 'reproduce' : x}
    
    encoder = (lambda: None, encoder_forward)
    
    decoder_heads = parallel_dict_layer({
        'forward' : layer_sequence((
            linear_layer(in_dim, 2, use_bias=True),
            #print_activations_layer('forward:'),
            categorical_sampler_layer(),
            #print_activations_layer('forward_sample:'),
        )),
        'rotate' : layer_sequence((
            linear_layer(in_dim, 3, use_bias=True),
            #print_activations_layer('rotate:'),
            categorical_sampler_layer(choices=jnp.array([-1,0,1])),
            #print_activations_layer('rotate_sample:'),
        )),
        'reproduce' : layer_sequence((
            linear_layer(in_dim, 2, use_bias=True),
            #print_activations_layer('reproduce:'),
            categorical_sampler_layer(),
            #print_activations_layer('reproduce_sample:'),
        )),
    })
    
    decoder = layer_sequence((
        decoder_heads,
        (lambda: None, lambda x : NomNomAction(**x)),
        #print_activations_layer('action:'),
    ))
    
    return layer_sequence([
        encoder,
        decoder,
    ])


def mutate(params, mutation_rate=0.1):
    """Apply Gaussian noise to parameters for mutation."""
    return tree_map(lambda p: p + mutation_rate * jrng.normal(jrng.PRNGKey(0), p.shape), params)


def crossover(parent1, parent2):
    """Blend two parent parameter sets to create a child."""
    return tree_map(lambda p1, p2: (p1 + p2) / 2, parent1, parent2)


def nomnom_model(params=NomNomModelParams()):
    
    # utility
    # - make a relu
    relu = (lambda: None, jnn.relu)
    
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
