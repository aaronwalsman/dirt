import os
from typing import Any, Optional, Tuple

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

from dirt.constants import DEFAULT_FLOAT_DTYPE
from dirt.envs.landscape import (
    LandscapeParams,
)
from dirt.envs.tera_arium import (
    TeraAriumParams,
    TeraAriumAction,
    TeraAriumTraits,
    tera_arium_renderer,
    tera_arium,
)
from dirt.visualization.viewer import Viewer

@commandline_interface
@static_dataclass
class TrainParams:
    seed : int = 1234
    initial_players : int = 2048
    max_players : int = 2048
    world_size : Tuple[int,int] = (512, 512)
    output_directory : str = '.'
    load_state : str = ''
    visualize : bool = False
    vis_width : int = 1024
    vis_height : int = 1024
    env_params : Any = TeraAriumParams(
        landscape = LandscapeParams(
            initial_total_energy = 128**2,
            mean_energy_sites = 128**2,
            initial_total_biomass = 128**2,
            mean_biomass_sites = 128**2,
            terrain_octaves = 12,
            terrain_unit_scale = 0.0025,
            terrain_max_height = 100.,
        )
    )
    train_params : Any = NaturalSelectionParams(
        max_population=max_players,
    )
    runner_params : Any = EpochRunnerParams(
        epochs=4,
        steps_per_epoch=1000,
        save_state=True,
        save_reports=True,
    )

@static_dataclass
class TrainReport:
    terrain : jnp.ndarray = False
    water : jnp.ndarray = False
    energy : jnp.ndarray = False
    biomass : jnp.ndarray = False
    light : jnp.ndarray = False
    
    players : jnp.ndarray = False
    player_x : jnp.ndarray = False
    player_r : jnp.ndarray = False
    
    sun_direction : jnp.ndarray = False
    
def log(epoch, reports):
    print(f'Epoch: {epoch}')
    #population_size = jnp.sum(reports.players[-1])
    #print(f'Population Size: {population_size}')
    # do other wandb stuff

def configure(params):
    def make_report(
        state, players, parents, children, actions, traits, adaptations
    ):
        
        #jax.debug.print('x {x}', x=state.env_state.bugs.x[:8])
        
        return TrainReport(
            terrain=state.env_state.landscape.terrain,
            water=state.env_state.landscape.water,
            energy=state.env_state.landscape.energy,
            biomass=state.env_state.landscape.biomass,
            light=state.env_state.landscape.light,
            players=players,
            player_x=state.env_state.bugs.x,
            player_r=state.env_state.bugs.r,
            #sun_direction=state.env_state.landscape.sundial.sun_direction,
        )
    
    render = tera_arium_renderer(params.env_params)
    
    def terrain_texture(report, texture_size):
        th, tw = texture_size
        terrain = report.terrain
        world_size = terrain.shape
        h, w = world_size
        assert th % h == 0
        assert tw % w == 0

        ry = th//h
        rx = tw//w
       
        print('Sun direction:', report.sun_direction)
        
        #texture = jnp.full((h, w), 127, dtype=jnp.uint8)
        #texture = jnp.repeat(texture, ry, axis=0)
        #texture = jnp.repeat(texture, rx, axis=1)
        #texture = jnp.repeat(texture[:,:,None], 3, axis=2)
        
        texture = render(
            report.water,
            report.temperature,
            report.energy,
            report.biomass,
            jnp.zeros((0,2), dtype=jnp.int32),
            jnp.zeros((0,3), dtype=jnp.int32),
            report.light,
        )
        print('MAX LIGHT:', jnp.max(report.light))
        return np.array((texture * 255).astype(jnp.uint8))

    def get_player_energy(params, report):
        return 1. #report.player_energy / params.env_params.max_energy
    
    return make_report, terrain_texture, get_player_energy

if __name__ == '__main__':

    # get the parameters from the commandline
    params = TrainParams().from_commandline(skip_overrides=True)
    params = params.override_descendants()
    
    make_report, terrain_texture, get_player_energy = configure(params)
    
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
            get_player_energy=None,
            get_terrain_map=lambda report : report.terrain + report.water,
            #get_water_map=lambda report : report.water,
            get_terrain_texture=terrain_texture,
            #get_sun_direction=lambda report : report.sun_direction,
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
            base_color = jnp.full(
                (state.shape[0], 3),
                jnp.array([1.,0.,0.], dtype=DEFAULT_FLOAT_DTYPE),
                dtype=DEFAULT_FLOAT_DTYPE
            ),
            photosynthesis = jnp.ones(state.shape[0]),
        )
        def model(key):
            forward_key, rotate_key, eat_key = jrng.split(key, 3)
            forward = jrng.choice(forward_key, jnp.array([0,1]))
            rotate = jrng.choice(rotate_key, jnp.array([-1,0,1]))
            eat = jrng.choice(eat_key, jnp.array([0,1]))
            return TeraAriumAction(
                forward=forward,
                rotate=rotate,
                bite=0,
                eat=1,
                reproduce=1,
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
            make_report,
            log,
            output_directory=params.output_directory,
            load_state=params.load_state,
        )
