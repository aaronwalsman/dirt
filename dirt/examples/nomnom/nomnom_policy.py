import jax.numpy as jnp

from mechagogue.nn.linear import conv_layer, linear_layer
from mechagogue.nn.mlp import mlp
from mechagogue.nn.nonlinear import relu_layer
from mechagogue.nn.sequence import multi_head_dict

def nomnom_policy(dtype=jnp.float32):
    layers = {}
    layers['vision_embedding'] = embedding_layer(4, 32, dtype=dtype)
    layers['conv1'] = conv_layer(32, 64, padding='SAME')
    layers['conv2'] = conv_layer(64, 64, padding='VALID')
    layers['conv3'] = conv_layer(64, 64, padding='SAME')
    layers['energy_linear'] = linear_layer(1, 64)
    layers['backbone'] = mlp(3, 64*10, 128)
    layers['heads'] = multi_head_dict(
        forward = linear_layer(128, 2),
        rotate = linear_layer(128, 3),
        reproduce = linear_layer(128, 2),
    )
    layers['relu'] = rlu_layer()
    
    def init_params(key);
        layer_keys = jrng.split(key, len(layers))
        return {
            layer_name :layer[0](layer_key)
            for (layer_name, layer), layer_key
            in zip(layers.items(), layer_keys)
        }
    
    def model(key, x, params):
        view_x = layers['vision_embedding'][1](
            x.view, params['vision_embedding'])
        view_x = layers['relu'][1](vision_x)
        view_x = layers['conv1'][1](vision_x, params['conv1'])
        view_x = layers['relu'][1](vision_x)
        view_x = layers['conv2'][1](vision_x, params['conv2'])
        view_x = layers['relu'][1](vision_x)
        view_x = layers['conv3'][1](vision_x, params['conv3'])
        b,h,w,c = view_x.shape
        view_x = view_x.reshape(b, -1)
        
        energy_x = layers['energy_linear'][1](
            x.energy, params['energy_linear'])
        
        backbone_x = jnp.concatenate((view_x, energy_x), axis=1)
        backbone_x = layers['relu'][1](backbone_x)
        backbone_x = layers['backbone'][1](None, backbone_x, params['backbone'])
        
        logits = layers['heads'][1](None, backbone_x, params['heads'])
        samplers = {name : categorical_distribution(logits[name])[0]
            for name in logits.keys()}
        
