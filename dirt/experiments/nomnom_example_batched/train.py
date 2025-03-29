import os
from typing import Any, Optional

import numpy as np

import jax.numpy as jnp
import jax.random as jrng

from tqdm import tqdm

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

@commandline_interface
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
    mutate = normal_mutate(learning_rate=3e-4)

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
    d = {}
    max_players = params.max_players
    seed = params.seed
    output_directory = './experiments/cool/'
    d["mean_initial_food"]=8**2
    d["max_initial_food"]=32**2
    d["mean_food_growth"]=2**2
    d["max_food_growth"]=16**2
    d["initial_players"]=32
    d["max_players"]=max_players
    d["world_size"]=(32,32)

    # Set specific value
    for k, v in m.items():
        if k in d:
            d[k] = v
        if k == 'seed':
            seed = v
        if k == 'output_directory':
            output_directory = v

    env_params = NomNomParams(
        mean_initial_food=d["mean_initial_food"],
        max_initial_food=d["max_initial_food"],
        mean_food_growth=d["mean_food_growth"],
        max_food_growth=d["max_food_growth"],
        initial_players=d["initial_players"],
        max_players=d["max_players"],
        world_size=d["world_size"]
    )
    train_params = TrainParams(seed=seed, max_players=max_players, output_directory=output_directory, env_params=env_params)
    return train_params

def run_experiment(key, params):
    if not os.path.exists(params.output_directory):
        os.makedirs(params.output_directory)
    save_leaf_data(params, f'{params.output_directory}/params.state')
    train(key, params)

if __name__ == '__main__':

    # get the parameters from the commandline
    params = TrainParams().from_commandline()
    
    experiment = {
      'name' : 'exp1_ip',
      'field' : "initial_players",
      'values' : jnp.arange(2, 18, 4),
      'prng_seeds' : [747, 1337, 520, 999, 1999, 123, 246, 8462, 9428, 1111]
    }

    # setup the key
    name = experiment['name']
    field = experiment['field']
    values = experiment['values']
    seeds = experiment['prng_seeds']
    for value in values:
        for i, seed in enumerate(seeds):
            print(f'\n{field}={value}: Run {i+1} / {len(seeds)}')
            output_directory = f'./experiments/{name}/{field}={value}/run{i}/'
            params = modify_params(params, {field: value, 'seed': seed, 'output_directory': output_directory})
            # bparams = modify_params(aparams, 'seed', seed)
            # cparams = modify_params(bparams, 'output_directory', output_directory)
            key = jrng.key(seed)
            run_experiment(key, params)