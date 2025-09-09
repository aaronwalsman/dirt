from typing import Any

import jax
import jax.random as jrng
import jax.numpy as jnp

from mechagogue.static import static_data
from mechagogue.nn.layer import make_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.static import static_functions
from mechagogue.nn.initializers import kaiming, zero
from mechagogue.nn.distributions import categorical_sampler_layer
from mechagogue.nn.nonlinear import relu_layer
from mechagogue.nn.linear import linear_layer, conv_layer
from mechagogue.nn.attention import make_attention_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.mlp import mlp

from dirt.envs.tera_arium import TeraAriumTraits
from dirt.constants import DEFAULT_FLOAT_DTYPE

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

default_backbone_channels = 256

def make_reshape_layer(output_shape):
    return make_layer(forward=lambda x : x.reshape(output_shape))

def make_nonvisual_encoder(
    include_audio,
    audio_channels,
    include_smell,
    smell_channels,
    include_wind,
    include_temperature,
    include_water,
    
    out_channels=default_backbone_channels,
    dtype=DEFAULT_FLOAT_DTYPE
):
    nonvisual_channels = 0
    nonvisual_channels += 1 # age
    nonvisual_channels += 1 # newborn
    nonvisual_channels += include_audio * audio_channels
    nonvisual_channels += include_smell * smell_channels
    nonvisual_channels += include_wind * 2 # wind
    nonvisual_channels += include_temperature * 1
    nonvisual_channels += include_water * 2
    nonvisual_channels += include_energy * 2
    nonvisual_channels += include_biomass * 2
    
    linear1 = linear_layer(
        in_channels=nonvisual_channels,
        out_channels=out_channels,
        include_bias=include_bias,
        dtype=dtype,
    )
    
    def forward(x, state):
        features = [
            x.age.flatten().astype(dtype),
            x.newborn.flatten().astype(dtype),
        ]
        if include_audio:
            features.append(x.audio.flatten().astype(dtype))
        if include_smell:
            features.append(x.smell.flatten().astype(dtype))
        if include_wind:
            features.append(x.wind.flatten().astype(dtype))
        if include_temperature:
            features.append(x.temperature.flatten().astype(dtype))
        if include_water:
            features.append(x.external_water.flatten().astype(dtype))
            features.append(x.internal_water.flatten().astype(dtype))
        if include_energy:
            features.append(x.external_energy.flatten().astype(dtype))
            features.append(x.internal_energy.flatten().astype(dtype))
        if include_biomass:
            features.append(x.external_biomass.flatten().astype(dtype))
            features.append(x.internal_biomass.flatten().astype(dtype))
        features.append(x.health.flatten().astype(dtype))
        x = jnp.concatenate(features)
        
        x = linear1.forward(x, state)
        
        return x

def make_flatten_vision_encoder(
    input_shape,
    in_channels=4,
    out_channels=default_backbone_channels,
    use_bias=True,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    flattened_channels = input_shape[0] * input_shape[1] * in_channels
    flatten_layer = make_reshape_layer(flattened_channels)
    linear1 = linear_layer(
        flattened_channels,
        out_channels,
        use_bias=use_bias,
        dtype=dtype,
    )
    
    return layer_sequence((flatten_layer, linear1))

def make_conv_flatten_vision_encoder(
    input_shape,
    in_channels=4,
    hidden_channels=64,
    out_channels=default_backbone_channels,
    use_bias=False,
    dtype=DEFAULT_FLOAT_DTYPE,
):
    assert input_shape[0] % 3 == 0
    assert input_shape[1] % 3 == 0
    
    conv1 = conv_layer(
        in_channels,
        hidden_channels,
        kernel_size=(3,3),
        stride=(3,3),
        padding='VALID',
        use_bias=use_bias,
        dtype=dtype,
    )
    
    post_conv_shape = (input_shape[0] // 3, input_shape[1] // 3)
    flattened_channels = (
        post_conv_shape[0] * post_conv_shape[1] * hidden_channels)
    flatten1 = make_reshape_layer((flattened_channels,))
    
    relu1 = relu_layer()
    
    linear1 = linear_layer(
        flattened_channels,
        out_channels,
        use_bias=use_bias,
        dtype=dtype,
    )
    
    return layer_sequence((conv1, flatten1, relu1, linear1))

def make_conv_attention_vision_encoder(
    input_shape,
    in_channels=4,
    hidden_channels=64,
    out_channels=default_backbone_channels,
    use_bias=False,
    dtype=DEFAULT_FLOAT_DTYPE,
    attention_mode='soft',
):
    
    assert input_shape[0] % 3 == 0
    assert input_shape[1] % 3 == 0
    
    tokens_h = input_shape[0] // 3
    tokens_w = input_shape[1] // 3
    total_tokens = tokens_h * tokens_w
    
    conv1 = conv_layer(
        in_channels,
        hidden_channels,
        kernel_size=(3,3),
        stride=(3,3),
        padding='VALID',
        use_bias=use_bias,
        dtype=dtype,
    )
    
    kv_linear = linear_layer(
        hidden_channels, hidden_channels*2, use_bias=use_bias, dtype=dtype)
    
    attention1 = make_attention_layer(1., attention_mode)
    
    relu1 = relu_layer()
    
    linear1 = linear_layer(
        hidden_channels,
        out_channels,
        use_bias=use_bias,
        dtype=dtype,
    )
    
    @static_data
    class ConvAttentionEncoderState:
        conv1 : Any
        kv_linear : Any
        position_embedding : Any
        q : Any
        linear1 : Any
    
    @static_functions
    class ConvAttentionEncoder:
        def init(key):
            conv1_key, kv_linear_key, pe_key, q_key, linear1_key = jrng.split(
                key, 5)
            conv1_state = conv1.init(conv1_key)
            kv_linear_state = kv_linear.init(kv_linear_key)
            position_embedding_state = jnp.zeros(
                (total_tokens, hidden_channels))
            q_state = jrng.normal(
                q_key, (1, hidden_channels)) * (1./hidden_channels)**0.5
            linear1_state = linear1.init(linear1_key)
            return ConvAttentionEncoderState(
                conv1=conv1_state,
                kv_linear=kv_linear_state,
                position_embedding=position_embedding_state,
                q=q_state,
                linear1=linear1_state,
            )
        
        def forward(key, x, state):
            x = conv1.forward(x, state.conv1)
            x = x.reshape(-1, hidden_channels) + state.position_embedding
            kv = kv_linear.forward(x, state.kv_linear)
            k = kv[...,:hidden_channels].reshape(-1, hidden_channels)
            v = kv[...,hidden_channels:].reshape(-1, hidden_channels)
            
            x = attention1.forward(key, (state.q, k, v))[0]
            x = relu1.forward(x)
            x = linear1.forward(x, state.linear1)
            return x
    
    return ConvAttentionEncoder
