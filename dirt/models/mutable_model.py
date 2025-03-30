import jax.numpy as jnp
import jax.random as jrng

from mechagogue.nn.linear import linear_layer
from mechagogue.nn.nonlinear import relu_layer
from mechagogue.nn.initializers import kaiming, zero
from mechagogue.tree import tree_getitem

def obsevation_encoder(dtype=jnp.bfloat16):
    def model(x):
        pass
    
    return lambda: None, model

def dynamic_channels_layer(
    initial_channels,
    max_channels,
):
    def init():
        return jnp.array([initial_channels], dtype=jnp.int32)
    
    def model(x, state):
        mask = jnp.arange(max_channels) < state
        x = x * mask
        return x
    
    return init, model

def backbone(
    initial_channels,
    max_channels,
    initial_layers,
    max_layers,
    use_bias=False,
    init_weights=kaiming,
    init_bias=zero,
    shared_dynamic_channels=True,
    dtype=jnp.bfloat16,
):
    _, model_relu = relu_layer()
    init_linear, model_linear = linear_layer(
        max_channels,
        max_channels,
        use_bias=use_bias,
        init_weights=init_weights,
        init_bias=init_bias,
        dtype=dtype,
    )
    init_dynamic_channels, model_dynamic_channels = dynamic_channels_layer(
        initial_channels, max_channels)
    
    def init(key):
        linear_keys = jrng.split(key, max_layers)
        linear_state = [init_linear(k) for k in linear_keys]
        if shared_dynamic_channels:
            dynamic_channel_state = init_dynamic_channels()
        else:
            dynamic_channel_state = [
                init_dynamic_channels() for _ in range(max_layers+1)]
        
        return linear_state, dynamic_channel_state
    
    def model(x, state):
        linear_state, dynamic_channel_state = state
        if shared_dynamic_channels:
            x = model_dynamic_channels(x, dynamic_channel_state)
        else:
            x = model_dynamic_channels(x, dynamic_channel_state[0])
        for i in range(max_layers):
            x = model_linear(x, linear_state[i])
            if shared_dynamic_channels:
                x = model_dynamic_channels(x, dynamic_channel_state)
            else:
                x = model_dynamic_channels(x, dynamic_channel_state[i+1])
            x = model_relu(x)
        
        return x
    
    return init, model

def mutate_backbone(
    min_channels,
    max_channels,
    weight_mutation_rate,
    bias_mutation_rate,
    channel_mutation_rate,
    shared_dynamic_channels=True,
    dtype=jnp.bfloat16,
):
    def mutate(key, state):
        linear_state, dynamic_channel_state = tree_getitem(state, 0)
        next_linear_state = []
        for weight, bias in linear_state:
            key, weight_key = jrng.split(key)
            weight_delta = jrng.normal(weight_key, weight.shape, dtype=dtype)
            weight = weight + weight_delta * weight_mutation_rate
            
            if bias is not None:
                key, bias_key = jrng.split(key)
                bias_delta = jrng.normal(bias_key, bias.shape, dtype=dtype)
                bias = bias + bias_delta * bias_mutation_rate
            
            next_linear_state.append((weight, bias))
        
        if shared_dynamic_channels:
            key, dynamic_channel_key = jrng.split(key)
            dynamic_channel_delta = jrng.choice(
                dynamic_channel_key,
                jnp.array([-1,0,1], dtype=jnp.int32),
                p=jnp.array([
                    channel_mutation_rate/2,
                    1.-channel_mutation_rate,
                    channel_mutation_rate/2.
                ], dtype=dtype),
            )
            dcs = dynamic_channel_state + dynamic_channel_delta
            dcs = jnp.clip(dcs, min_channels, max_channels)
            next_dynamic_channel_state = dcs
        else:
            next_dynamic_channel_state = []
            for dcs in dynamic_channel_state:
                key, dynamic_channel_key = jrng.split(key)
                dynamic_channel_delta = jrng.choice(
                    dynamic_channel_key,
                    jnp.array([-1,0,1], dtype=jnp.int32),
                    p=jnp.array([
                        channel_mutation_rate/2,
                        1.-channel_mutation_rate,
                        channel_mutation_rate/2.
                    ], dtype=dtype),
                )
                dcs += dynamic_channel_delta
                dcs = jnp.clip(dcs, min_channels, max_channels)
                next_dynamic_channel_state.append(dcs)
        
        return next_linear_state, next_dynamic_channel_state
    
    return mutate

def virtual_parameters(state):
    linear_state, dynamic_channel_state = state
    c = 0
    for in_c, out_c in zip(
        dynamic_channel_state[:-1], dynamic_channel_state[1:]
    ):
        c += in_c * out_c
    
    return c

def init_test():
    key = jrng.key(1238)
    h = 256
    
    key, init_key = jrng.split(key)
    k = kaiming(init_key, (h,h))
    
    key, x_key = jrng.split(key)
    x = jrng.normal(x_key, (h,)) / (h**0.5)
    
    kx = k @ x
    kxr = kx * (kx > 0.)
    print(kx.shape)
    print(jnp.linalg.norm(x))
    print(jnp.linalg.norm(kxr))
    
    g = kaiming(init_key, (1000, h, h))
    sum_g = g.sum(axis=0)
    print(jnp.linalg.norm(g[0].reshape(-1)))
    print(jnp.linalg.norm(sum_g.reshape(-1)))
    print(1000**0.5 * 22.57)

if __name__ == '__main__':
    init_test()
