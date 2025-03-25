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

from dirt.envs.nomnom import NomNomParams, NomNomAction, nomnom
from dirt.models.nomnom import (
    NomNomModelParams, nomnom_unconditional_or_linear_population)
from dirt.visualization.viewer import Viewer

'''
4 : 16
6 : 64
8 : 256
10 : 1024
12 : 4096
'''
size = 10
exp_size = 2**size

@commandline_interface
@static_dataclass
class TrainParams:
    seed : int = 1234
    max_players : int = (exp_size//4)**2
    output_directory : str = '.'
    load_state : str = ''
    visualize : bool = False
    vis_width : int = 1024
    vis_height : int = 1024
    env_params : Any = NomNomParams(
        mean_initial_food=(exp_size//2)**2,
        max_initial_food=(exp_size//2)**2,
        mean_food_growth=max((exp_size/24.)**2,2),
        max_food_growth=(exp_size//8)**2,
        initial_players=(exp_size//4),
        max_players=max_players,
        world_size=(exp_size,exp_size)
    )
    train_params : Any = NaturalSelectionParams(
        max_population=max_players,
    )
    runner_params : Any = EpochRunnerParams(
        epochs=100,
        steps_per_epoch=100,
        save_state=False,
        save_reports=True,
    )

#@static_dataclass
#class TrainReport:
#    actions : Any = NomNomAction(0,0,0)
#    players : jnp.array = False
#    player_x : jnp.array = False
#    player_r : jnp.array = False
#    player_energy : jnp.array = False
#    food_grid : jnp.array = False
#    player_type : jnp.array = False

def make_report(
    state,
    players,
    parents,
    children,
    actions,
    traits,
    adaptations,
):
    #return TrainReport(
    #    actions,
    #    players,
    #    state.env_state.player_x,
    #    state.env_state.player_r,
    #    state.env_state.player_energy,
    #    state.env_state.food_grid,
    #    state.model_state[0]
    #)
    population_0 = jnp.sum((state.model_state[0] == 0) & players)
    population_1 = jnp.sum((state.model_state[0] == 1) & players)
    return (population_0, population_1)

def log(epoch, reports):
    #population_size = jnp.sum(reports.players[-1])
    print(f'Epoch: {epoch}')
    #print(f'Population Size: {population_size}')
    
    #zero_type = (reports.player_type[-1] == 0) & reports.players[-1]
    #one_type = (reports.player_type[-1] == 1) & reports.players[-1]
    #print(f'Unconditional: {jnp.sum(zero_type)}')
    #print(f'Linear: {jnp.sum(one_type)}')
    zero_type = reports[0][-1]
    one_type = reports[1][-1]
    print(f'Unconditional: {zero_type}')
    print(f'Linear: {one_type}')
    
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
   
    #texture = food_grid.astype(jnp.uint8) * 128 + 127
    #texture = jnp.repeat(texture, ry, axis=0)
    #texture = jnp.repeat(texture, rx, axis=1)
    #texture = jnp.repeat(texture[:,:,None], 3, axis=2)
    
    background_color = jnp.array([255, 240, 212], dtype=jnp.uint8)
    #food_color = jnp.array([163,214,115], dtype=jnp.uint8)
    #food_color = jnp.array([96,163,84], dtype=jnp.uint8)
    food_color = jnp.array([124,177,94], dtype=jnp.uint8)
    food_grid_r = jnp.repeat(food_grid, ry, axis=0)
    food_grid_r = jnp.repeat(food_grid_r, rx, axis=1)
    
    texture = jnp.where(food_grid_r[...,None], food_color, background_color)
    
    return np.array(texture)

def get_player_energy(params, report):
    return report.player_energy / params.env_params.max_energy

def get_player_color(player_id, report):
    if report.player_type[player_id]:
        return (0.25,0.25,1)
    else:
        return (1,0.25,0.25)

if __name__ == '__main__':
        
    # get the parameters from the commandline
    params = TrainParams().from_commandline()
    params = params.replace(
        env_params=params.env_params.replace(max_players=params.max_players)
    )
    params = params.replace(
        train_params=params.train_params.replace(
            max_population=params.max_players)
    )
    
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
            terrain_texture_multiple=4,
            get_player_energy=get_player_energy,
            get_terrain_texture=terrain_texture,
            get_player_color=get_player_color,
        )
        viewer.begin()
    
    else:
        for seed in (0,1,2,3,4,5,6,7):
            print(f'seed: {seed}')
            params = params.replace(
                seed=seed,
                output_directory=f'exp_{size}_{seed}',
            )
            
            if not os.path.exists(params.output_directory):
                os.makedirs(params.output_directory)
            save_leaf_data(params, f'{params.output_directory}/params.state')

            # setup the key
            key = jrng.key(params.seed)

            # build the nomnom environment
            reset_env, step_env = nomnom(params.env_params)
            #mutate = normal_mutate(learning_rate=3e-4)

            # build the model
            model_params = NomNomModelParams(
                view_width=params.env_params.view_width,
                view_distance=params.env_params.view_distance,
            )
            #init_model, model = nomnom_unconditional_model(model_params)
            init_population, player_traits, model, mutate, adapt = (
                nomnom_unconditional_or_linear_population(model_params, 3e-4))
            
            # build the trainer
            init_train, step_train = natural_selection(
                params.train_params,
                reset_env,
                step_env,
                init_population,
                player_traits,
                model,
                mutate,
                adapt,
            )

            # run
            breakpoint()
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
