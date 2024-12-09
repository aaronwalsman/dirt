import time

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.pop.objective_free import objective_free, ObjectiveFreeParams

from dirt.examples.nomnom import nomnom, NomNomParams, NomNomAction

def train(key, iterations):

    # build the environment
    env_params = NomNomParams(mean_food_growth=1.5)
    reset_env, step_env = nomnom(env_params)

    def randomized_policy(weights, obs):
        n = weights.shape[0]
        
        # super clunky, streamline pls
        def sampler(key):
            #keys = jrng.split(key, n)
            return NomNomAction.uniform_sample(key, n)
        
        return sampler, lambda x : 0

    # build the algorithm
    algo_params = ObjectiveFreeParams()
    reset_algo, step_algo = objective_free(
        algo_params,
        reset_env,
        step_env,
        randomized_policy,
        lambda key : 0,
        lambda x : x
    )

    key, reset_key = jrng.split(key)
    algo_state = reset_algo(reset_key)
    
    def step(algo_state, key):
        env_state, _, players, _, _ = algo_state
        #jax.debug.print('Population: {p}', p=jnp.sum(players != -1))
        #jax.debug.print('Food remaining: {f}', f=jnp.sum(env_state.food_grid))
        #jax.debug.print('OP: {p}', p=players)
        #jax.debug.print('Players: {p}', p=env_state.player_id)
        #jax.debug.print('Parents: {p}', p=env_state.parent_id)
        return step_algo(key, *algo_state)
    
    key, step_key = jrng.split(key)
    step_keys = jrng.split(step_key, iterations)
    algo_state, _ = jax.lax.scan(step, algo_state, step_keys, iterations)
    
    return algo_state

if __name__ == '__main__':
    train = jax.jit(train, static_argnums=(1))

    key = jrng.key(1234)
    iterations = 50

    t = time.time()
    things = train(key, iterations)
    print((time.time() - t)/iterations)
    #breakpoint()
