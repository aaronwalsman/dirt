'''
This example is designed to test the nomnom environment in a training loop
with a random policy.
'''

import time

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.pop.natural_selection import (
    natural_selection, NaturalSelectionParams)
from mechagogue.breed.normal import normal_mutate
from mechagogue.nn.mlp import mlp

from dirt.examples.nomnom.nomnom_env import nomnom, NomNomParams, NomNomAction
from dirt.examples.nomnom.nomnom_model import NomNomModelConfig, nomnom_model

def train(key, env_params, train_params, iterations):

    # build the environment
    reset_env, step_env = nomnom(env_params)
    
    # build the randomized policy
    def randomized_policy(weights, obs):
        n = weights.shape[0]
        
        # super clunky, streamline pls
        def sampler(key):
            #keys = jrng.split(key, n)
            return NomNomAction.uniform_sample(key, n)
        
        return sampler, lambda x : 0
    
    # build mutate
    mutate = normal_mutate(learning_rate=3e-4)
    
    #init_mlp, model_mlp = mlp(hidden_layers=4,
    #    in_channels=256,
    #    hidden_channels=32,
    #    out_channels=16)
    model_config = NomNomModelConfig(
        view_width=env_params.view_width,
        view_distance=env_params.view_distance,
    )
    init_model, model = nomnom_model(model_config)
    
    # build the natural selection algorithm
    reset_train, step_train = natural_selection(
        train_params,
        reset_env,
        step_env,
        init_model,
        model,
        mutate,
    )
    
    # reset the training algorithm to get an initial state
    key, reset_key = jrng.split(key)
    train_state, active_players = reset_train(reset_key)
    
    steps_per_epoch = 1000
    # step function that will be run at each iteration
    # this is just the algorithm's step function, but structured so that it
    # can be scanned
    def train_block(train_state_active, key):
        # import pdb; pdb.set_trace()
        train_state, active_players = train_state_active
        env_state = train_state.env_state
        jax.debug.print(
            'Population: {p}, Food: {f}',
            p=jnp.sum(active_players != -1),
            f=jnp.sum(env_state.food_grid)
        )
        
        steps_per_epoch = 100
        
        def scan_body(train_state_active, key):
            train_state, _ = train_state_active
            next_train_state, active_players, parents, children = step_train(
                key, train_state)
            return (
                (next_train_state, active_players),
                (active_players, parents, children),
            )
        
        train_state, _ = jax.lax.scan(
            scan_body,
            (train_state, active_players),
            jrng.split(key, steps_per_epoch),
        )
        return train_state, None
    
    # generate step keys
    key, step_key = jrng.split(key)
    step_keys = jrng.split(step_key, iterations)
    
    # iterate
    train_state, _ = jax.lax.scan(
        train_block,
        (train_state, active_players),
        step_keys,
    )
    
    return train_state

if __name__ == '__main__':
    train = jax.jit(train, static_argnums=(1,2,3))
    
    '''
    env_params = NomNomParams(
        mean_initial_food=1500**2,
        max_initial_food=4000**2,
        mean_food_growth=150**2,
        max_food_growth=1000**2,
        initial_players=1000,
        max_players=5000000, # this is very slow on laptop
        world_size=(10000,10000)
    )
    '''
    max_players = 1024
    env_params = NomNomParams(
        mean_initial_food=16**2,
        max_initial_food=32**2,
        mean_food_growth=4**2,
        max_food_growth=16**2,
        initial_players=32,
        max_players=max_players,
        world_size=(32,32)
    )
    algo_params = NaturalSelectionParams(max_players)

    key = jrng.key(1234)
    iterations = 100

    t = time.time()
    result = train(key, env_params, algo_params, iterations)
    jax.block_until_ready(result)
    algo_fps = iterations / (time.time() - t)
    print('algo fps:', algo_fps)
    print('env fps:', algo_params.rollout_steps * algo_fps)
