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

from dirt.examples.nomnom.nomnom_env import NomNomObservation, NomNomAction

@static_dataclass
class NomNomModelConfig:
    num_input_classes : int = 4
    view_width : int = 5
    view_distance : int = 5

def nomnom_model(config=NomNomModelConfig()):
    
    # make a relu
    relu = (lambda: None, jnn.relu)
    
    # view encoder
    view_embedding = embedding_layer(config.num_input_classes, 32)
    view_conv1 = conv_layer(32, 64, padding='VALID')
    view_conv2 = conv_layer(64, 128, padding='VALID')
    flattened_channels = (config.view_width-4) * (config.view_distance-4) * 128
    flatten = (lambda: None, lambda x : x.reshape(flattened_channels))
    view_fc1 = linear_layer(flattened_channels, 128)
    view_encoder = layer_sequence(
        (view_embedding, view_conv1, relu, view_conv2, flatten, relu, view_fc1)
    )
    
    # energy encoder
    energy_encoder = mlp(
        hidden_layers=1,
        in_channels=1,
        hidden_channels=32,
        out_channels=128,
    )
    
    # combine the encoders
    encoder = layer_sequence((
        (lambda: None, lambda x : {'view': x.view, 'energy': x.energy}),
        parallel_dict_layer({'view':view_encoder, 'energy':energy_encoder}),
        (lambda: None, lambda x : sum(x.values())),
        relu,
    ))
    
    # backbone
    backbone = layer_sequence((
        linear_layer(128, 64),
        relu,
        linear_layer(64, 32),
        relu,
        (lambda: None, lambda x : {'forward':x, 'rotate':x, 'reproduce':x}),
    ))
    
    # decoder heads
    decoder_heads = parallel_dict_layer({
        'forward' : layer_sequence(
            (linear_layer(32, 2), categorical_sampler_layer)),
        'rotate' : layer_sequence(
            (linear_layer(32, 3), categorical_sampler_layer)),
        'reproduce' : layer_sequence(
            (linear_layer(32, 2), categorical_sampler_layer)),
    })
    
    decoder = layer_sequence((
        decoder_heads,
        (lambda: None, lambda x : NomNomAction(**x))
    ))
    
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
