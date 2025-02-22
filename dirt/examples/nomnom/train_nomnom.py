'''
This example is designed to test the nomnom environment in a training loop
with a random policy.
'''
import time
import argparse
from typing import Any, Optional

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.pop.natural_selection import (
    natural_selection, NaturalSelectionParams)
from mechagogue.breed.normal import normal_mutate
from mechagogue.nn.mlp import mlp
from mechagogue.static_dataclass import static_dataclass
from mechagogue.serial import save_leaf_data, load_from_example

from dirt.examples.nomnom.nomnom_env import nomnom, NomNomParams, NomNomAction
from dirt.examples.nomnom.nomnom_model import NomNomModelParams, nomnom_model

import wandb

@static_dataclass
class NomNomTrainParams:
    max_players : int = 256
    env_params : Any = NomNomParams(
        mean_initial_food=8**2,
        max_initial_food=32**2,
        mean_food_growth=2**2,
        max_food_growth=16**2,
        initial_players=32,
        max_players=max_players,
        world_size=(32,32)
    )
    train_params : Any = NaturalSelectionParams(
        max_population=max_players,
    )
    epochs : int = 100
    steps_per_epoch : int = 1000
    output_directory : str = './'
    load_from_file : Optional[str] = None

def train(key, params):
    wandb.init(project="nomnom",
               entity="harvardml"
               )
    
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
    
    # get the initial state of the training function
    key, reset_key = jrng.split(key)
    train_state, _ = jax.jit(reset_train)(reset_key)
    epoch = 0
    if params.load_from_file is not None:
        key, epoch, train_state = load_like(
            (key, epoch, train_state), params.load_from_file)
    
    def make_report(
        state, actions, next_state, players, parent_locations, child_locations
    ):
        player_x = next_state.env_state.player_x
        player_r = next_state.env_state.player_r
        
        return {
            'player_x' : player_x,
            'player_r' : player_r,
        }
    
    # precompile the primary epoch train computation
    def train_epoch(epoch_key, train_state):
        def scan_body(train_state, step_key):
            next_train_state, report = step_train(step_key, train_state)
            return next_train_state, report
        
        train_state, reports = jax.lax.scan(
            scan_body,
            train_state,
            jrng.split(epoch_key, params.steps_per_epoch),
            logging_info
        )
        
        return train_state, reports
    train_epoch = jax.jit(train_epoch)
    
    save_leaf_data(
        train_params,
        f'{params.output_directory}/train_params.state',
    )
    
    # the outer loop is not scanned because it will have side effects
    while epoch < params.epochs:
        print(f'Epoch: {epoch}')
        key, epoch_key = jrng.split(key)
        env_state = train_state.env_state
        
        train_state, reports = train_epoch(epoch_key, train_state)
        
        save_leaf_data(
            (key, epoch, train_state),
            f'{params.output_directory}/train_state_{epoch}.state',
        )
        save_leaf_data(
            reports,
            f'{params.output_directory}/report_{epoch}.state',
        )
        epoch += 1
        
        train_state, active_players, logging_info = train_epoch(
            epoch_key, train_state, active_players, logging_info)
        
        # For the 3 components of actions
        wandb.log(logging_info, step=train_state)
        wandb.log(logging_info, step=train_state)
        wandb.log(logging_info, step=train_state)
        
        import pdb; pdb.set_trace()
    
    return train_state

if __name__ == '__main__':
    
    key = jrng.key(1234)
    
    max_players = 256
    env_params = NomNomParams(
        mean_initial_food=8**2,
        max_initial_food=32**2,
        mean_food_growth=2**2,
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
        steps_per_epoch=100,
    )
    
    # update these defaults with commandline arguments
    parser = argparse.ArgumentParser()
    params.add_commandline_args(parser)
    args = parser.parse_args()
    params = params.from_commandline_args(args)

    train(key, params)
