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
from mechagogue.static_dataclass import static_dataclass
from mechagogue.serial import save_leaf_data, load_example_data

#from dirt.examples.nomnom.nomnom_env import nomnom, NomNomParams, NomNomAction
from dirt.envs.nomnom import nomnom, NomNomParams, NomNomAction
from dirt.examples.nomnom.nomnom_model import (
    NomNomModelParams, nomnom_model, nomnom_linear_model)

import wandb
import matplotlib.pyplot as plt

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
    output_directory : str = '/n/holylfs06/LABS/kempner_fellow_awalsman/Lab/chloe00/linear_model'
    load_from_file : Optional[str] = None

@static_dataclass
class NomNomReport:
    actions : Any = NomNomAction(0,0,0)
    players : jnp.array = False
    player_x : jnp.array = False
    player_r : jnp.array = False
    player_energy : jnp.array = False
    food_grid : jnp.array = False

def make_report(
    state, actions, next_state, players, parent_locations, child_locations
):
    #jax.debug.print('player_x {x}', x=next_state.env_state.player_x)
    #jax.debug.print('player_r {r}', r=next_state.env_state.player_r)
    #jax.debug.print('action {a}', a=actions)
    return NomNomReport(
        actions,
        players,
        next_state.env_state.player_x,
        next_state.env_state.player_r,
        next_state.env_state.player_energy,
        next_state.env_state.food_grid,
    )

def train(key, params):
    #wandb.init(project="nomnom",
    #           entity="harvardml"
    #           )
    
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
    #init_model, model = nomnom_model(model_params)
    init_model, model = nomnom_linear_model(model_params)
    
    # - build the training functions
    reset_train, step_train = natural_selection(
        params.train_params,
        reset_env,
        step_env,
        init_model,
        model,
        mutate,
        make_report
    )
    
    # get the initial state of the training function
    key, reset_key = jrng.split(key)
    train_state, _ = jax.jit(reset_train)(reset_key)
    epoch = 0
    if params.load_from_file is not None:
        key, epoch, train_state = load_example_data(
            (key, epoch, train_state), params.load_from_file)
    
    # precompile the primary epoch train computation
    def train_epoch(epoch_key, train_state):
        def scan_body(train_state, step_key):
            next_train_state, report = step_train(step_key, train_state)
            return next_train_state, report
        
        train_state, reports = jax.lax.scan(
            scan_body,
            train_state,
            jrng.split(epoch_key, params.steps_per_epoch),
        )
        
        return train_state, reports
    train_epoch = jax.jit(train_epoch)
    
    save_leaf_data(
        params,
        f'{params.output_directory}/train_params.state',
    )
    
    # the outer loop is not scanned because it will have side effects
    while epoch < params.epochs:
        print(f'Epoch: {epoch}')
        key, epoch_key = jrng.split(key)
        
        train_state, reports = train_epoch(epoch_key, train_state)
        
        save_leaf_data(
            (key, epoch, train_state),
            f'{params.output_directory}/train_state_{epoch:08}.state',
        )
        save_leaf_data(
            reports,
            f'{params.output_directory}/report_{epoch:08}.state',
        )
        epoch += 1
        
        #actions, players = reports
        actions = reports.actions
        players = reports.players
        # For the 3 components of actions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].hist(jnp.mean(actions.forward * players, axis=0), bins=20, color="blue", alpha=0.7)
        axes[0].set_title("Action distribution 1: forward")

        axes[1].hist(jnp.mean(actions.rotate * players, axis=0), bins=20, color="red", alpha=0.7)
        axes[1].set_title("Action distribution 2: rotate")

        axes[2].hist(jnp.mean(actions.reproduce * players, axis=0), bins=20, color="green", alpha=0.7)
        axes[2].set_title("Action distribution 3: reproduce")

        fig.tight_layout()
        #wandb.log({"plot/actions": wandb.Image(fig)})
        #wandb.log({"active players": players.sum()})
        
    
    return train_state

if __name__ == '__main__':
    
    #key = jrng.key(5432)
    key = jrng.key(1234)
    
    max_players = 512#*16
    env_params = NomNomParams(
        max_energy=4.,
        mean_initial_food=100000, #8**2,
        max_initial_food=100000, #32**2,
        mean_food_growth=16, #16*16, #2**2,
        max_food_growth=1000, #16**2,
        initial_players=8,
        max_players=max_players,
        world_size=(64,64),#(512,512),
        senescence=0.01,
    )
    train_params = NaturalSelectionParams(
        max_population=max_players,
    )
    params = NomNomTrainParams(
        env_params=env_params,
        train_params=train_params,
        epochs=830,
        steps_per_epoch=2048,
    )
    
    # update these defaults with commandline arguments
    parser = argparse.ArgumentParser()
    params.add_commandline_args(parser)
    args = parser.parse_args()
    params = params.update_from_commandline(args)
    
    start = time.time()
    train(key, params)
    print(time.time() - start)
