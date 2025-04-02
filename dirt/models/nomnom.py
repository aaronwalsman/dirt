import jax
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
from mechagogue.breed.normal import normal_mutate

from dirt.envs.nomnom import NomNomObservation, NomNomAction, NomNomTraits

# utilities
relu = (lambda: None, jnn.relu)

def nomnom_linear_observation_encoder(dtype=jnp.float32):
    def model(x):
        food = (x.view == 1).reshape(-1).astype(dtype)
        players = (x.view == 2).reshape(-1).astype(dtype)
        out_of_bounds = (x.view == 3).reshape(-1).astype(dtype)
        return jnp.concatenate(
            (food, players, out_of_bounds, x.energy[...,None]), axis=-1)
    
    return lambda: None, model

def nomnom_action_decoder(in_channels, use_bias=True, dtype=jnp.float32):
    return layer_sequence((
        (
            lambda: None,
            lambda x : {'forward' : x, 'rotate' : x, 'reproduce' : x}
        ),
        parallel_dict_layer({
            'forward' : layer_sequence((
                linear_layer(in_channels, 2, use_bias=use_bias, dtype=dtype),
                categorical_sampler_layer(),
            )),
            'rotate' : layer_sequence((
                linear_layer(in_channels, 3, use_bias=use_bias, dtype=dtype),
                categorical_sampler_layer(choices=jnp.array([-1,0,1])),
            )),
            'reproduce' : layer_sequence((
                linear_layer(in_channels, 2, use_bias=use_bias, dtype=dtype),
                categorical_sampler_layer(),
            )),
        }),
        (lambda: None, lambda x : (NomNomAction(**x), None)),
    ))

# params
@static_dataclass
class NomNomModelParams:
    num_input_classes : int = 4
    view_width : int = 5
    view_distance : int = 5

# models
def nomnom_unconditional_model(params=NomNomModelParams(), dtype=jnp.float32):
    return layer_sequence((
        (lambda : None, lambda x : jnp.ones((1,), dtype=dtype)),
        nomnom_action_decoder(1, dtype=dtype),
    ))

def nomnom_linear_model(params=NomNomModelParams(), dtype=jnp.float32):
    in_dim = (
        (params.num_input_classes-1) *
        params.view_width *
        params.view_distance +
        1   # energy
    )
    
    encoder = nomnom_linear_observation_encoder(dtype=dtype)
    decoder = nomnom_action_decoder(in_dim, dtype=dtype)
    
    return layer_sequence((encoder, decoder))

def init_uniform_population_params(init_model_params):
    init_model_params = ignore_unused_args(init_model_params, ('key',))
    init_model_params = jax.vmap(init_model_params)
    def init(key, population_size, max_population_size):
        init_keys = jrng.split(key, max_population_size)
        return init_model_params(init_keys)

def nomnom_unconditional_or_linear_population(
    params=NomNomModelParams(),
    learning_rate=3e-4,
    dtype=jnp.bfloat16,
):
    in_dim = (
        (params.num_input_classes-1) *
        params.view_width *
        params.view_distance +
        1   # energy
    )
    linear_encoder = nomnom_linear_observation_encoder(dtype=dtype)
    linear_decoder = nomnom_action_decoder(in_dim, dtype=dtype)
    init_linear_model, linear_model = layer_sequence(
        (linear_encoder, linear_decoder))
    init_linear_model = jax.vmap(init_linear_model)
    
    unconditional_encoder = (
        lambda : None, lambda x : jnp.ones((1,), dtype=dtype))
    unconditional_decoder = nomnom_action_decoder(
        1, use_bias=False, dtype=dtype)
    init_unconditional_model, unconditional_model = layer_sequence(
        (unconditional_encoder, unconditional_decoder))
    init_unconditional_model = jax.vmap(init_unconditional_model)
    
    mutator = normal_mutate(learning_rate)
    
    def init(key, population_size, max_population_size):
        init_keys = jrng.split(key, 2*max_population_size)
        linear_keys = init_keys[:max_population_size]
        unconditional_keys = init_keys[max_population_size:]
        linear_state = init_linear_model(linear_keys)
        unconditional_state = init_unconditional_model(unconditional_keys)
        
        # 50/50
        model_type = jnp.arange(max_population_size) < (population_size//2)
        model_type = model_type.astype(jnp.int32)
        # all unconditional
        #model_type = jnp.zeros(max_population_size, dtype=jnp.int32)
        # all linear
        #model_type = jnp.ones(max_population_size, dtype=jnp.int32)
        
        return (model_type, unconditional_state, linear_state)
    
    def player_traits(model_state):
        return NomNomTraits()
    
    def model(key, x, state):
        model_type, unconditional_state, linear_state = state
        unconditional_key, linear_key = jrng.split(key)
        unconditional_action = unconditional_model(
            unconditional_key, x, unconditional_state)
        linear_action = linear_model(
            linear_key, x, linear_state)
        
        def select_leaf(a, b):
            return jnp.where(model_type, a, b)
        return jax.tree.map(select_leaf, linear_action, unconditional_action)
    
    def adapt(model_state):
        return model_state
    
    def mutate(key, state):
        model_type, unconditional_state, linear_state = state
        unconditional_state, linear_state = mutator(
            key, (unconditional_state, linear_state))
        return (model_type[0], unconditional_state, linear_state)
    
    return init, player_traits, model, mutate, adapt

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
