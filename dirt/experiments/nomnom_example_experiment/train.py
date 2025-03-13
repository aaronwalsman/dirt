import os
from typing import Any, Optional

import numpy as np

import jax.numpy as jnp
import jax.random as jrng

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
    load_state : str = None
    visualize : bool = False
    vis_width : int = 1024
    vis_height : int = 1024
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
        epochs=100,
        steps_per_epoch=1000,
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

def log(reports):
    population_size = jnp.sum(reports.players[-1])
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

if __name__ == '__main__':

    # get the parameters from the commandline
    params = TrainParams().from_commandline()
    
    if params.visualize:
        # get the path to the params and reports
        params_path = f'{params.output_directory}/params.state'
        report_paths = sorted([
            f'{params.output_directory}/{file_path}'
            for file_path in os.listdir(params.output_directory)
            if file_path.startswith('report') and file_path.endswith('.state')
        ])
       
        # launch the viewer
        viewer = Viewer(
            TrainParams(),
            params_path,
            TrainReport(),
            report_paths,
            window_width=params.vis_width,
            window_height=params.vis_height,
            get_player_energy=get_player_energy,
            get_terrain_texture=terrain_texture,
        )
        viewer.begin()
    
    else:
        if not os.path.exists(params.output_directory):
            os.makedirs(params.output_directory)
        save_leaf_data(params, f'{params.output_directory}/params.state')

        # setup the key
        key = jrng.key(params.seed)

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
