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

from dirt.examples.nomnom import nomnom, NomNomParams, NomNomAction

def train(key, env_params, algo_params, iterations):

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

    # build the natural selection algorithm
    reset_algo, step_algo = natural_selection(
        algo_params,
        reset_env,
        step_env,
        randomized_policy,
        lambda key : 0,
        lambda w, players, parents, next_players, next_parents : w
    )
    
    # reset the algorithm to get an initial state
    key, reset_key = jrng.split(key)
    algo_state = reset_algo(reset_key)
    
    # step function that will be run at each iteration
    # this is just the algorithm's step function, but structured so that it
    # can be scanned
    def step(algo_state, key):
        env_state, _, players, _, _ = algo_state
        jax.debug.print(
            'Population: {p}, Food: {f}',
            p=jnp.sum(players != -1),
            f=jnp.sum(env_state.food_grid)
        )
        return step_algo(key, *algo_state)
    
    # generate step keys
    key, step_key = jrng.split(key)
    step_keys = jrng.split(step_key, iterations)
    
    # iterate
    algo_state, _ = jax.lax.scan(step, algo_state, step_keys, iterations)
    
    return algo_state

if __name__ == '__main__':
    train = jax.jit(train, static_argnums=(1,2,3))
    
    env_params = NomNomParams(
        mean_initial_food=1500**2,
        max_initial_food=4000**2,
        mean_food_growth=150**2,
        max_food_growth=1000**2,
        initial_players=1000,
        max_players=5000000, # this is very slow on laptop
        world_size=(10000,10000)
    )
    algo_params = NaturalSelectionParams()

    key = jrng.key(1234)
    iterations = 100

    t = time.time()
    result = train(key, env_params, algo_params, iterations)
    jax.block_until_ready(result)
    algo_fps = iterations / (time.time() - t)
    print('algo fps:', algo_fps)
    print('env fps:', algo_params.rollout_steps * algo_fps)
