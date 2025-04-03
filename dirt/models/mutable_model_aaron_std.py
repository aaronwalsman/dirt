import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.nn.linear import linear_layer
from mechagogue.nn.nonlinear import relu_layer
from mechagogue.nn.initializers import kaiming, kaiming_std, zero
from mechagogue.tree import tree_getitem

def obsevation_encoder(dtype=jnp.bfloat16):
    def model(x):
        pass
    
    return lambda: None, model

def mutable_channels_layer(
    initial_channels,
    max_channels,
):
    def init():
        return jnp.array(initial_channels, dtype=jnp.int32)
    
    def model(x, state):
        mask = jnp.arange(max_channels) < state
        x = x * mask
        return x
    
    return init, model

def mutable_switch_layer(
    initial_value,
):
    def init():
        return jnp.array(initial_value, dtype=jnp.bool)
    
    def model(x, state):
        return x * state
    
    return init, model

def init_mutable_linear_weight(initial_in_channels, initial_out_channels):
    def init(key, shape, dtype):
        std = kaiming_std(initial_in_channels)
        weight = jrng.normal(key, shape, dtype=dtype) * std
        in_mask = jnp.arange(shape[-2]) < initial_in_channels
        out_mask = jnp.arange(shape[-1]) < initial_out_channels
        weight = weight * in_mask[:,None]
        weight = weight * out_mask
        
        return weight
    
    return init

def get_mutable_weight_mean_std(weight, in_channels, out_channels):
    *b,inc,outc = weight.shape
    weight_sum = weight.sum(axis=-1).sum(axis=-1)
    weight_mean = weight_sum / (in_channels * out_channels)
    weight_var = (weight_mean[...,None,None] - weight)**2
    in_mask = jnp.arange(inc) < in_channels
    # breakpoint()
    out_mask = jnp.arange(outc) < out_channels
    weight_var = weight_var * in_mask[:,None] * out_mask[None,:]
    weight_var_sum = weight_var.sum(axis=-1).sum(axis=-1)
    weight_var_mean = weight_var_sum / (in_channels * out_channels)
    weight_std = jnp.sqrt(weight_var_mean)
    
    return weight_mean, weight_std

def mutable_mlp(
    in_channels,
    out_channels,
    initial_hidden_channels,
    max_hidden_channels,
    initial_hidden_layers,
    max_hidden_layers,
    use_bias=False,
    dtype=jnp.bfloat16,
):
    
    _, model_relu = relu_layer()
    
    init_in_linear, model_in_linear = linear_layer(
        in_channels,
        max_hidden_channels,
        use_bias=use_bias,
        init_weights=init_mutable_linear_weight(
            in_channels, initial_hidden_channels),
        init_bias=zero,
        dtype=dtype,
    )
    
    init_hidden_linear, model_hidden_linear = linear_layer(
        max_hidden_channels,
        max_hidden_channels,
        use_bias=use_bias,
        init_weights=init_mutable_linear_weight(
            initial_hidden_channels, initial_hidden_channels),
        init_bias=zero,
        dtype=dtype,
    )
    
    init_out_linear, model_out_linear = linear_layer(
        max_hidden_channels,
        out_channels,
        use_bias=use_bias,
        init_weights=init_mutable_linear_weight(
            initial_hidden_channels, out_channels),
        init_bias=zero,
        dtype=dtype,
    )
    
    init_mutable_channels, model_mutable_channels = mutable_channels_layer(
        initial_hidden_channels, max_hidden_channels)
    
    init_mutable_switch_on, model_mutable_switch = mutable_switch_layer(
        jnp.ones((), dtype=jnp.bool))
    init_mutable_switch_off, _ = mutable_switch_layer(
        jnp.zeros((), dtype=jnp.bool))
    
    def init(key):
        
        state = {}
        
        key, in_key = jrng.split(key)
        in_linear_state = init_in_linear(in_key)
        state['in_linear'] = in_linear_state
        
        key, hidden_key = jrng.split(key)
        hidden_keys = jrng.split(hidden_key, max_hidden_layers)
        hidden_linear_states = []
        mutable_switch_states = []
        for i, hidden_key in enumerate(hidden_keys):
            hidden_linear_state = init_hidden_linear(hidden_key)
            hidden_linear_states.append(hidden_linear_state)
            
            if i < initial_hidden_layers:
                mutable_switch_state = init_mutable_switch_on()
            else:
                mutable_switch_state = init_mutable_switch_off()
            mutable_switch_states.append(mutable_switch_state)
            
        state['hidden_linear'] = hidden_linear_states
        state['mutable_switch'] = mutable_switch_states
        
        key, out_key = jrng.split(key)
        out_linear_state = init_out_linear(out_key)
        state['out_linear'] = out_linear_state
        
        mutable_channels_state = init_mutable_channels()
        state['mutable_channels'] = mutable_channels_state
        
        return state
    
    def model(x, state):
        
        x = model_in_linear(x, state['in_linear'])
        x = model_mutable_channels(x, state['mutable_channels'])
        
        for i in range(max_hidden_layers):
            x0 = x
            x1 = model_hidden_linear(x0, state['hidden_linear'][i])
            x1 = model_mutable_channels(x1, state['mutable_channels'])
            x1 = model_relu(x1)
            x1 = model_mutable_switch(x1, state['mutable_switch'][i])
            x = x0 + x1
        
        x = model_out_linear(x, state['out_linear'])
        
        return x
    
    return init, model

def mutate_mutable_linear(
    key,
    linear_state,
    in_channels,
    out_channels,
    switch,
    weight_mutation_rate,
    bias_mutation_rate,
):
    weight, bias = linear_state
    
    if in_channels is None:
        in_channels = weight.shape[-2]
    if out_channels is None:
        out_channels = weight.shape[-1]
    
    # apply a random offset in weight space
    key, weight_key = jrng.split(key)
    weight_delta = jrng.normal(weight_key, weight.shape, dtype=weight.dtype)
    decay = (1 - (weight_mutation_rate**2)*in_channels/2)**0.5
    
    mode = 'std'
    if mode == 'decay':
        weight = weight * decay + weight_delta * weight_mutation_rate
    elif mode == 'std':
        weight = weight + weight_delta * weight_mutation_rate
        mean, std = get_mutable_weight_mean_std(
            weight, in_channels, out_channels)
        #jax.debug.print('mean {m}, std {s}', m=mean, s=std)
        weight = (weight - mean)/(std+1e-8) * kaiming_std(in_channels)
    
    # zero out unused channels
    max_channels_in = weight.shape[-2]
    max_channels_out = weight.shape[-1]
    in_channel_mask = jnp.arange(max_channels_in) < in_channels
    out_channel_mask = jnp.arange(max_channels_out) < out_channels
    weight = weight * in_channel_mask[..., None]
    weight = weight * out_channel_mask
    
    # if the switch is off, zero out the entire weight matrix
    weight = jnp.where(switch, weight, jnp.zeros_like(weight))
    
    # apply a random offset to the bias
    if bias is not None:
        key, bias_key = jrng.split(key)
        bias_delta = jrng.normal(bias_key, bias.shape, dtype=bias.dtype)
        bias = bias + bias_delta * bias_mutation_rate
        bias = bias * out_channel_mask
        
        # if the switch is off, zero out the entire bias vector
        bias = jnp.where(switch, bias, jnp.zeros_like(bias))
    
    return weight, bias

def mutate_mutable_mlp(
    min_channels,
    max_channels,
    weight_mutation_rate,
    bias_mutation_rate,
    channel_mutation_rate,
    switch_mutation_rate,
):
    
    def mutate(key, state):
        
        next_state = {}
        
        key, mutable_channel_key = jrng.split(key)
        mutable_channel_delta = jrng.choice(
            mutable_channel_key,
            jnp.array([-1,0,1], dtype=jnp.int32),
            p=jnp.array([
                channel_mutation_rate/2,
                1.-channel_mutation_rate,
                channel_mutation_rate/2.
            ], dtype=jnp.bfloat16),
        )
        mutable_channels_state = (
            state['mutable_channels'] + mutable_channel_delta)
        mutable_channels_state = jnp.clip(
            mutable_channels_state, min_channels, max_channels)
        next_state['mutable_channels'] = mutable_channels_state
        
        key, in_key = jrng.split(key)
        weight, bias = state['in_linear']
        weight, bias = mutate_mutable_linear(
            in_key,
            (weight, bias),
            None,
            mutable_channels_state,
            jnp.array(1, dtype=jnp.bool),
            weight_mutation_rate,
            bias_mutation_rate,
        )
        next_state['in_linear'] = (weight, bias)
        
        hidden_linear_states = []
        mutable_switch_states = []
        for i, (weight, bias) in enumerate(state['hidden_linear']):
            
            key, switch_key = jrng.split(key)
            flip = jrng.bernoulli(switch_key, switch_mutation_rate)
            mutable_switch_state = jnp.logical_xor(
                state['mutable_switch'][i], flip)
            mutable_switch_states.append(mutable_switch_state)
            
            key, hidden_key = jrng.split(key)
            weight, bias = mutate_mutable_linear(
                hidden_key,
                (weight, bias),
                mutable_channels_state,
                mutable_channels_state,
                mutable_switch_state,
                weight_mutation_rate,
                bias_mutation_rate,
            )
            hidden_linear_states.append((weight, bias))
        
        next_state['hidden_linear'] = hidden_linear_states
        next_state['mutable_switch'] = mutable_switch_states
        
        key, out_key = jrng.split(key)
        weight, bias = state['out_linear']
        weight, bias = mutate_mutable_linear(
            out_key,
            (weight, bias),
            mutable_channels_state,
            None,
            jnp.array(1, dtype=jnp.bool),
            weight_mutation_rate,
            bias_mutation_rate,
        )
        next_state['out_linear'] = (weight, bias)
        
        return next_state
    
    return mutate

# DEPRECATED BELOW HERE

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
        
        next_linear_state = []
        for i, (weight, bias) in enumerate(linear_state):
            
            # get in_channels and out_channels
            if shared_dynamic_channels:
                in_channels = next_dynamic_channel_state[0]
                out_channels = next_dynamic_channel_state[0]
            else:
                in_channels = next_dynamic_channel_state[i][0]
                out_channels = next_dynamic_channel_state[i+1][0]
            
            key, weight_key = jrng.split(key)
            weight_delta = jrng.normal(weight_key, weight.shape, dtype=dtype)
            decay = (1 - (weight_mutation_rate**2)*in_channels/2)**0.5
            #decay = 1.
            weight = weight * decay + weight_delta * weight_mutation_rate
            
            # update weights
            in_channel_mask = jnp.arange(max_channels) < in_channels
            out_channel_mask = jnp.arange(max_channels) < out_channels
            weight = weight * in_channel_mask[..., None]
            weight = weight * out_channel_mask
            
            if bias is not None:
                key, bias_key = jrng.split(key)
                bias_delta = jrng.normal(bias_key, bias.shape, dtype=dtype)
                bias = bias + bias_delta * bias_mutation_rate
                bias = bias * out_channel_mask
            
            next_linear_state.append((weight, bias))
        
        return next_linear_state, next_dynamic_channel_state
    
    return mutate

def backbone_weight_info(model_state, shared_dynamic_channels):
    linear_state, dynamic_channel_state = model_state
    weight_info = []
    for i, layer_state in enumerate(linear_state):
        if shared_dynamic_channels:
            in_channels = dynamic_channel_state[...,0]
            out_channels = dynamic_channel_state[...,0]
        else:
            in_channels = dynamic_channel_state[i][0]
            out_channels = dynamic_channel_state[i+1][0]
        *b,inc,outc = layer_state[0].shape
        weight_sum = layer_state[0].sum(axis=-1).sum(axis=-1)
        weight_mean = weight_sum / (in_channels * out_channels)
        weight_var = weight_mean[...,None,None] - layer_state[0]
        in_mask = jnp.arange(inc) < in_channels[:,None]
        out_mask = jnp.arange(outc) < out_channels[:,None]
        weight_var = weight_var * in_mask[:,:,None] * out_mask[:,None,:]
        weight_var_sum = weight_var.sum(axis=-1).sum(axis=-1)
        weight_var_mean = weight_var_sum / (in_channels * out_channels)
        weight_std = jnp.sqrt(weight_var_mean)
        weight_info.append({'mean':weight_mean, 'std':weight_std})
        
    return weight_info

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
