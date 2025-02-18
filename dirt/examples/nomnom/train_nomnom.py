'''
This example is designed to test the nomnom environment in a training loop
with a random policy.
'''

import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.pop.natural_selection import (
    natural_selection, NaturalSelectionParams)
from mechagogue.breed.normal import normal_mutate
from mechagogue.nn.mlp import mlp
from mechagogue.static_dataclass import static_dataclass

from dirt.examples.nomnom.nomnom_env import nomnom, NomNomParams, NomNomAction
from dirt.examples.nomnom.nomnom_model import NomNomModelParams, nomnom_model

@static_dataclass
class NomNomTrainParams:
    env_params : Any
    train_params : Any
    epochs : int = 100
    steps_per_epoch : int = 1000

def train(key, params):
    
    # build the necessary functions
    # - build the environment functions
    reset_env, step_env = nomnom(params.env_params)
    
    # - build mutate function
    mutate = normal_mutate(learning_rate=3e-4)
    
    # - build the model functions
    model_params = NomNomModelParams(
        view_width=params.env_params.view_width,
        view_distance=params.env_params.view_distance,
    )
    init_model, model = nomnom_model(model_params)
    
    # - build the training functions
    reset_train, step_train = natural_selection(
        params.train_params,
        reset_env,
        step_env,
        init_model,
        model,
        mutate,
    )
    
    # - reset the training algorithm to get an initial state
    key, reset_key = jrng.split(key)
    train_state, active_players = reset_train(reset_key)
    
    # the outer loop is not scanned because it will have side effects
    # (saving checkpoints)
    #def train_block(train_state_active, key):
    for epoch in range(params.epochs):
        env_state = train_state.env_state
        jax.debug.print(
            'Population: {p}, Food: {f}',
            p=jnp.sum(active_players != -1),
            f=jnp.sum(env_state.food_grid)
        )
        
        def scan_body(train_state_active, key):
            train_state, _ = train_state_active
            next_train_state, active_players, parents, children = step_train(
                key, train_state)
            return (
                (next_train_state, active_players),
                (active_players, parents, children),
            )
        
        train_state_active_players, trajectories = jax.lax.scan(
            scan_body,
            (train_state, active_players),
            jrng.split(key, params.steps_per_epoch),
        )
        train_state, active_players = train_state_active_players
        
        # DUMP TRAJECTORIES HERE
    
    # generate step keys
    key, step_key = jrng.split(key)
    step_keys = jrng.split(step_key, params.epochs)
    
    # iterate
    train_state, _ = jax.lax.scan(
        train_block,
        (train_state, active_players),
        step_keys,
    )
    
    return train_state

if __name__ == '__main__':
    
    key = jrng.key(1234)
    
    max_players = 256
    env_params = NomNomParams(
        mean_initial_food=4**2,
        max_initial_food=32**2,
        mean_food_growth=1**2,
        max_food_growth=16**2,
        initial_players=32,
        max_players=max_players,
        world_size=(32,32)
    )
    train_params = NaturalSelectionParams(
        max_population=max_players,
    )
    params = NomNomTrainParams(
        env_params=env_params,
        train_params=train_params,
        epochs=10,
        steps_per_epoch=1,
    )

    train(key, params)
