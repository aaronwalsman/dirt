import time

import jax
import jax.numpy as jnp
import jax.random as jrng

import dirt.variable_length as vl

def test_ns_compact(
    key,
    start_players=int(5e3),
    max_players=int(5e6),
    p_reproduce=0.1,
    p_die=0.1,
    num_steps=10000,
):
    # initialize player ids
    player_id = jnp.arange(max_players)
    player_id = player_id.at[start_players:].set(-1)
    next_player_id = start_players
    
    # initialize player data
    key, data_key = jrng.split(key)
    player_data = jrng.uniform(
        data_key, minval=0., maxval=1., shape=(max_players, 8, 128, 128))
    
    def single_loop(state, key):
        player_data, player_id, next_player_id = state
        
        reproduce = jrng.uniform(
            key, minval=0, maxval=1, shape=(max_players,)) < p_reproduce
        reproduce = reproduce & (player_id != -1)
        n_reproduce = jnp.sum(reproduce)
        child_id = jnp.full(max_players, -1)
        reproduce_ids, = jnp.nonzero(
            reproduce, size=max_players, fill_value=max_players)
        child_id.at[reproduce_ids].set(
            jnp.arange(max_players) + next_player_id)
        child_data = player_data.at[reproduce_ids].get(
            mode='fill', fill_value=0.)
        
        player_id, player_data = vl.concatenate(
            (player_id, child_id), (player_data, child_data))
        
        next_player_id = next_player_id + n_reproduce
        
        return (player_data, player_id, next_player_id), None
    
    key, step_key = jrng.split(key)
    step_keys = jrng.split(step_key, num_steps)
    (player_data, player_id, next_player_id), _ = jax.lax.scan(
        single_loop,
        (player_data, player_id, next_player_id),
        step_keys,
    )
    
    return player_data, player_id, next_player_id

if __name__ == '__main__':
    jit_test_ns_compact = jax.jit(test_ns_compact, static_argnums=(1,2,3,4,5))
    
    # warmup
    warmup_player_data, _, _ = jit_test_ns_compact(
        jrng.key(1233),
        max_players=int(5e2),
        num_steps=10,
    )
    warmup_player_data.block_until_ready()
    
    t0 = time.time()
    player_data, player_id, next_player_id = jit_test_ns_compact(
        jrng.key(1234),
        max_players=int(5e4),
        num_steps=10,
    )
    player_data.block_until_ready()
    t1 = time.time()
    print(t1-t0)
