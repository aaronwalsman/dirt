import os
from typing import Any, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

from splendor.image import save_image

from mechagogue.static_dataclass import static_dataclass
from mechagogue.commandline import commandline_interface
from mechagogue.pop.natural_selection import (
    NaturalSelectionParams, natural_selection)
from mechagogue.epoch_runner import EpochRunnerParams, epoch_runner
from mechagogue.serial import save_leaf_data
from mechagogue.breed.normal import normal_mutate

from mechagogue.tree import tree_getitem

from dirt.constants import DEFAULT_FLOAT_DTYPE
from dirt.gridworld2d.weather import WeatherParams
from dirt.envs.landscape import (
    LandscapeParams,
)
from dirt.envs.tera_arium import (
    TeraAriumParams,
    TeraAriumAction,
    TeraAriumTraits,
    #render_tera_arium,
    tera_arium_renderer,
    tera_arium,
)
from dirt.visualization.viewer import Viewer

@commandline_interface
@static_dataclass
class TrainParams:
    seed : int = 1235
    initial_players : int = 2048
    max_players : int = 2048
    world_size : Tuple[int,int] = (256,256) #(1024,1024)
    output_directory : str = '.'
    load_state : str = ''
    visualize : bool = False
    vis_width : int = 1024
    vis_height : int = 1024
    downsample_visualizer : int = 1
    max_render_players : int =256
    env_params : Any = TeraAriumParams(
        landscape = LandscapeParams(
            initial_total_energy = 256**2,
            mean_energy_sites = 256**2,
            initial_total_biomass = 256**2,
            mean_biomass_sites = 256**2,
            terrain_octaves = 12,
            terrain_unit_scale = 0.0025,
            terrain_max_height = 200.,
            terrain_bias = -25,
            weather = WeatherParams(
                mountain_temperature_baseline = -3.,
                include_rain = False,
                include_temperature = False,
                include_wind = False,
            )
        )
    )
    train_params : Any = NaturalSelectionParams(
        max_population=max_players,
    )
    runner_params : Any = EpochRunnerParams(
        epochs=1,
        steps_per_epoch=1000,
        save_state=True,
        save_reports=True,
    )

@static_dataclass
class TrainReport:
    terrain : jnp.ndarray = False
    water : jnp.ndarray = False
    moisture : jnp.ndarray = False
    rain : jnp.ndarray = False
    temperature : jnp.ndarray = False
    energy : jnp.ndarray = False
    biomass : jnp.ndarray = False
    light : jnp.ndarray = False
    
    players : jnp.ndarray = False
    player_x : jnp.ndarray = False
    player_r : jnp.ndarray = False
    
    sun_direction : jnp.ndarray = False
    
    moisture_start_raining : float = 0.
    wind_direction : jnp.ndarray = False
    normalized_altitude : jnp.ndarray = False
    
def configure_functions(params):
    
    render_tera_arium = tera_arium_renderer(params.env_params)
    
    def make_report(
        state, players, parents, children, actions, traits, adaptations
    ):
        
        #jax.debug.print('x {x}', x=state.env_state.bugs.x[:8])
        
        altitude = (
            state.env_state.landscape.terrain + 
            state.env_state.landscape.water
        )
        normalized_altitude = jnp.clip(
            altitude/params.env_params.landscape.max_effective_altitude, 0., 1.)
        
        dv = params.downsample_visualizer
        moisture = state.env_state.landscape.moisture
        rain = state.env_state.landscape.rain
        temperature = state.env_state.landscape.temperature
        if params.env_params.landscape.weather.include_rain:
            moisture = moisure[::dv,::dv]
            rain = rain[::dv,::dv]
        if params.env_params.landscape.weather.include_temperature:
            temperature = temperature[::dv,::dv]
        return TrainReport(
            terrain=state.env_state.landscape.terrain[::dv,::dv],
            water=state.env_state.landscape.water[::dv,::dv],
            moisture=moisture,
            rain=rain,
            temperature=temperature,
            energy=state.env_state.landscape.energy[::dv,::dv],
            biomass=state.env_state.landscape.biomass[::dv,::dv],
            light=state.env_state.landscape.light[::dv,::dv],
            players=players,
            player_x=state.env_state.bugs.x,
            player_r=state.env_state.bugs.r,
            moisture_start_raining=params.env_params.landscape.weather.moisture_start_raining,
            wind_direction=state.env_state.landscape.wind,
            #sun_direction=state.env_state.landscape.sundial.sun_direction,
            normalized_altitude=normalized_altitude[::dv,::dv],
        )

    def log(epoch, reports):
        print(f'Epoch: {epoch}')
        #population_size = jnp.sum(reports.players[-1])
        #print(f'Population Size: {population_size}')
        # do other wandb stuff

    def terrain_texture(report, texture_size, display_mode):
        th, tw = texture_size
        terrain = report.terrain
        world_size = terrain.shape
        h, w = world_size
        assert th % h == 0
        assert tw % w == 0

        ry = th//h
        rx = tw//w
        
        if display_mode == 1:
            texture = render_tera_arium(
                report.water,
                report.temperature,
                report.energy,
                report.biomass,
                jnp.zeros((0,2), dtype=jnp.int32),
                jnp.zeros((0,3), dtype=jnp.int32),
                report.light,
            )
        
        elif display_mode == 2:
            texture = render_tera_arium(
                report.water,
                report.temperature,
                report.energy,
                report.biomass,
                jnp.zeros((0,2), dtype=jnp.int32),
                jnp.zeros((0,3), dtype=jnp.int32),
                jnp.ones_like(report.light),
            )
       
        elif display_mode == 3:
            if params.env_params.landscape.weather.include_temperature:
                temperature = report.temperature[...,None]
                hot = jnp.array([0.5, 0., 0.], dtype=temperature.dtype)
                cold = jnp.array([0., 0., 0.5], dtype=temperature.dtype)
                texture = jnp.where(
                    temperature >= 0.,
                    temperature * hot,
                    -temperature * cold,
                )
            else:
                texture = jnp.zeros((*terrain.shape,3), dtype=terrain.dtype)
            
        elif display_mode == 4:
            if params.env_params.landscape.weather.include_rain:
                moisture = report.moisture[...,None]
                normalized_moisture = (moisture/report.moisture_start_raining)
                texture = normalized_moisture * jnp.array([1.,1.,1.])
                rain_color = jnp.array([0.25, 0.25, 1.], dtype=moisture.dtype)
                texture = texture + report.rain[...,None] * rain_color
            else:
                texture = jnp.zeros((*terrain.shape,3), dtype=terrain.dtype)
        
        elif display_mode == 5:
            texture = (
                report.normalized_altitude[...,None] * jnp.array([1.,1.,1.]))
        
        else:
            texture = jnp.zeros((*params.world_size, 3))
        
        print('Wind direction', report.wind_direction)
        
        texture = np.array((texture * 255).astype(jnp.uint8))
        #save_image(texture, 'tmp.png')
        return texture
    
    return make_report, log, terrain_texture

def get_player_energy(params, report):
    return 1. #report.player_energy / params.env_params.max_energy

if __name__ == '__main__':

    # get the parameters from the commandline
    params = TrainParams().from_commandline(skip_overrides=True)
    params = params.override_descendants()
        
    make_report, log, terrain_texture = configure_functions(params)
    
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
            downsample_heightmap=params.downsample_visualizer,
            max_render_players=params.max_render_players,
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
