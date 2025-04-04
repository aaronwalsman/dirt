import os
from typing import Any, Optional

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static_dataclass import static_dataclass
from mechagogue.commandline import commandline_interface
from mechagogue.pop.natural_selection import (
    NaturalSelectionParams, natural_selection)
from mechagogue.epoch_runner import EpochRunnerParams, epoch_runner
from mechagogue.serial import save_leaf_data
from mechagogue.breed.normal import normal_mutate

from mechagogue.tree import tree_getitem

from dirt.envs.tera_arium import (
    TeraAriumParams, TeraAriumAction, TeraAriumTraits, tera_arium)
from dirt.visualization.viewer import Viewer

@commandline_interface
@static_dataclass
class TrainParams:
    seed : int = 1234
    initial_players : int = 32
    max_players : int = 256
    output_directory : str = '.'
    load_state : str = ''
    visualize : bool = False
    vis_width : int = 1024
    vis_height : int = 1024
    env_params : Any = TeraAriumParams(
        world_size=(32,32),
    )
    train_params : Any = NaturalSelectionParams(
        max_population=max_players,
    )
    runner_params : Any = EpochRunnerParams(
        epochs=100,
        steps_per_epoch=1000,
        save_state=True,
        save_reports=True,
    )

@static_dataclass
class TrainReport:
    terrain : jnp.ndarray
    water : jnp.ndarray

def make_reporter(params):
    def make_report(
        state, players, parents, children, actions, traits, adaptations
    ):
        return TrainReport(
            terrain=state.env_state.landscape.terrain,
            water=state.env_state.landscape.water,
        )
    
    return make_report

def log(epoch, reports):
    print(f'Epoch: {epoch}')
    #population_size = jnp.sum(reports.players[-1])
    #print(f'Population Size: {population_size}')
    # do other wandb stuff

def terrain_texture(report, texture_size):
    th, tw = texture_size
    terrain = report.terrain
    world_size = terrain.shape
    h, w = world_size
    assert th % h == 0
    assert tw % w == 0

    ry = th//h
    rx = tw//w
   
    #texture = food_grid.astype(jnp.uint8) * 128 + 127
    texture = jnp.full((h, w), 127, dtype=jnp.uint8)
    texture = jnp.repeat(texture, ry, axis=0)
    texture = jnp.repeat(texture, rx, axis=1)
    texture = jnp.repeat(texture[:,:,None], 3, axis=2)
    return np.array(texture)

def get_player_energy(params, report):
    return 1. #report.player_energy / params.env_params.max_energy

if __name__ == '__main__':

    # get the parameters from the commandline
    params = TrainParams().from_commandline()
    params = params.override_descendants()
    
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
        reset_env, step_env = tera_arium(params.env_params)
        #mutate = normal_mutate(learning_rate=3e-4)
        breed = lambda state : tree_getitem(state, 0)
        adapt = lambda state : state

        # build the model
        # - temporary random model
        init_population = lambda max_population_size : jnp.ones(
            max_population_size)
        model_traits = lambda state : TeraAriumTraits(
            body_size = jnp.ones(state.shape[0]),
            brain_size = jnp.ones(state.shape[0]),
            photosynthesis = jnp.ones(state.shape[0]),
        )
        def model(key):
            forward_key, rotate_key = jrng.split(key)
            forward = jrng.choice(forward_key, jnp.array([0,1]))
            rotate = jrng.choice(rotate_key, jnp.array([-1,0,1]))
            return TeraAriumAction(
                forward,
                rotate,
                0,
                0,
                0,
            ), None
        
        # build the trainer
        init_train, step_train = natural_selection(
            params.train_params,
            reset_env,
            step_env,
            init_population,
            model_traits,
            model,
            breed,
            adapt,
        )
        
        # run
        epoch_runner(
            key,
            params.runner_params,
            init_train,
            step_train,
            make_reporter(params),
            log,
            output_directory=params.output_directory,
            load_state=params.load_state,
        )
