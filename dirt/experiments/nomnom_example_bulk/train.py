import os
from typing import Any, Optional

import numpy as np

import jax.numpy as jnp
import jax.random as jrng

from tqdm import tqdm

from dataclasses import asdict

from mechagogue.static_dataclass import static_dataclass
from mechagogue.commandline import commandline_interface
from mechagogue.pop.natural_selection import (
    NaturalSelectionParams, natural_selection)
from mechagogue.epoch_runner import EpochRunnerParams, epoch_runner
from mechagogue.serial import save_leaf_data
from mechagogue.breed.normal import normal_mutate

from dirt.envs.nomnom import NomNomParams, NomNomAction, nomnom
from dirt.models.nomnom import NomNomModelParams, nomnom_linear_model
from dirt.visualization.viewer import Viewer

@static_dataclass
class TrainParams:
    seed : int = 1234
    max_players : int = 256
    output_directory : str = '.'
    load_state : str = ''
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
    runner_params : Any = EpochRunnerParams(
        epochs=4,
        steps_per_epoch=100,
        save_state=True,
        save_reports=True,
    )
    mutation_rate : float = 3e-4

@commandline_interface
@static_dataclass
class ExperimentParams:
    seed: int = 1234
    num_seeds : int = 5
    name : str = 'default'
    field : str = 'initial_players'
    value_start : float = 2
    value_end : float = 16
    num_values : int = 4
    train_params : Any = TrainParams()

@static_dataclass
class TrainReport:
    actions : Any = NomNomAction(0,0,0)
    players : jnp.array = False
    player_x : jnp.array = False
    player_r : jnp.array = False
    player_energy : jnp.array = False
    food_grid : jnp.array = False

def make_report(state, players, parents, children, actions):
    return TrainReport(
        actions,
        players,
        state.env_state.player_x,
        state.env_state.player_r,
        state.env_state.player_energy,
        state.env_state.food_grid,
    )

def log(epoch, reports):
    population_size = jnp.sum(reports.players[-1])
    print(f'Epoch: {epoch}')
    print(f'Population Size: {population_size}')
    # do other wandb stuff

def terrain_texture(report, texture_size):
    th, tw = texture_size
    food_grid = report.food_grid
    world_size = food_grid.shape
    h, w = world_size
    assert th % h == 0
    assert tw % w == 0

    ry = th//h
    rx = tw//w
   
    texture = food_grid.astype(jnp.uint8) * 128 + 127
    texture = jnp.repeat(texture, ry, axis=0)
    texture = jnp.repeat(texture, rx, axis=1)
    texture = jnp.repeat(texture[:,:,None], 3, axis=2)
    return np.array(texture)

def get_player_energy(params, report):
    return report.player_energy / params.env_params.max_energy

def train(key, params):
    # build the nomnom environment
    reset_env, step_env = nomnom(params.env_params)
    mutate = normal_mutate(learning_rate=params.mutation_rate)

    # build the model
    model_params = NomNomModelParams(
        view_width=params.env_params.view_width,
        view_distance=params.env_params.view_distance,
    )
    init_model, model = nomnom_linear_model(model_params)

    # build the trainer
    init_train, step_train = natural_selection(
        params.train_params, reset_env, step_env, init_model, model, mutate)
    
    # run
    epoch_runner(
        key,
        params.runner_params,
        init_train,
        step_train,
        make_report,
        log,
        output_directory=params.output_directory,
        load_state=params.load_state,
    )

def modify_params(params, m):
    # Set default values
    params_dict = asdict(params)
    train_params_dict = asdict(params.env_params)
    max_players = params.max_players
    seed = params.seed
    output_directory = params.output_directory
    mutation_rate = params.mutation_rate

    # Set specific value
    for k, v in m.items():
        if k in train_params_dict:
            train_params_dict[k] = v
        if k == 'max_players':
            max_players = v
        if k == 'seed':
            seed = v
        if k == 'output_directory':
            output_directory = v
        if k == 'mutation_rate':
            mutation_rate = v

    env_params = NomNomParams(**train_params_dict)
    train_params = TrainParams(seed=seed, max_players=max_players, output_directory=output_directory, env_params=env_params)
    return train_params

def run_experiment(key, params):
    if not os.path.exists(params.output_directory):
        os.makedirs(params.output_directory)
    save_leaf_data(params, f'{params.output_directory}/params.state')
    train(key, params)

if __name__ == '__main__':
    
    exp_params = ExperimentParams().from_commandline()
    params = exp_params.train_params;

    # setup the key
    values = jnp.linspace(exp_params.value_start, exp_params.value_end, exp_params.num_values)
    # OVERRIDE VALUES IF YOU WANT
    # values = jnp.array([3e-4, 3e-3, 3e-2, 3e-1])
    seeds = jrng.permutation(jrng.PRNGKey(exp_params.seed), jnp.arange(0, 1000))[:exp_params.num_seeds]
    for value in values:
        for i, seed in enumerate(seeds):
            print(f'\n{exp_params.field}={value}: Run {i+1} / {len(seeds)}')
            output_directory = f'./experiments/{exp_params.name}/{exp_params.field}={value}/run{i}/'
            params = modify_params(params, {exp_params.field: value, 'seed': seed, 'output_directory': output_directory})
            key = jrng.key(seed)
            run_experiment(key, params)