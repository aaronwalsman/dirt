import time

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.commandline import commandline_interface
from mechagogue.static import static_functions, static_data
from mechagogue.serial import save_leaf_data

from dirt.gridworld2d.landscape import LandscapeParams, make_landscape
from dirt.constants import DEFAULT_FLOAT_DTYPE
from dirt.visualization.viewer import Viewer

@commandline_interface
@static_data
class LandscapeExampleParams:
    seed : int = 1234
    
    output_directory : str = '.'
    visualize : bool = False
    
    landscape_params : LandscapeParams = LandscapeParams()
    
    steps = 1000

@static_data
class LandscapeExampleReport:
    rock : jnp.ndarray = False
    water : jnp.ndarray = False
    temperature : jnp.ndarray = False
    energy : jnp.ndarray = False
    biomass : jnp.ndarray = False
    light : jnp.ndarray = False

if __name__ == '__main__':
    
    float_dtype = DEFAULT_FLOAT_DTYPE
    
    params = LandscapeExampleParams().from_commandline()
    landscape_params = params.landscape_params.replace(
        #include_light=False,
        #include_wind=False,
        #include_temperature=False,
        #include_rain=False,
        #include_resources=False,
        include_smell=False,
        include_audio=False,
        fill_water_to_sea_level=True,
        #initial_water_per_cell=1.,
    )
    
    key = jrng.key(params.seed)
    
    landscape = make_landscape(landscape_params, float_dtype=float_dtype)
    
    params_path = f'{params.output_directory}/params.state'
    reports_path = f'{params.output_directory}/report.state'
    
    if params.visualize:
        key, init_key = jrng.split(key)
        
        def get_texture(report, texture_size, display_mode):
            th, tw = texture_size
            h, w = params.landscape_params.world_size
            assert h % th == 0
            assert w % tw == 0
            
            if display_mode == 1:
                texture = landscape.render(report, 2)
            
            return texture
        
        def get_terrain(report):
            return (report.rock + report.water) / (
                params.landscape_params.terrain_downsample**2)
        
        viewer = Viewer(
            LandscapeExampleParams(),
            params_path,
            LandscapeExampleReport(),
            [reports_path],
            get_terrain_map = get_terrain,
            get_active_players = None,
            get_terrain_texture = get_texture,
            get_water_map = None,
            get_sun_direction = None,
            downsample_heightmap = params.landscape_params.terrain_downsample,
        )
        viewer.begin()
    
    else:
        
        key, init_key = jrng.split(key)
        landscape_state = landscape.init(init_key)
        
        def landscape_step(key_state, _):
            key, state = key_state
            key, step_key = jrng.split(key)
            
            state = landscape.step(step_key, state)
            report = LandscapeExampleReport(
                rock=state.rock,
                water=state.water,
                temperature=state.temperature,
                energy=state.energy,
                biomass=state.biomass,
                light=state.light,
            )
            
            return (key, state), report
        
        @jax.jit
        def run_scan(key, landscape_state, steps):
            key_state, reports = jax.lax.scan(
                landscape_step,
                (key, landscape_state),
                None,
                length=params.steps,
            )
            
            return key_state, reports
        
        (key, landscape_state), reports = run_scan(
            key, landscape_state, params.steps)
        reports.water.block_until_ready()
        
        t0 = time.time()
        (key, landscape_state), reports = run_scan(
            key, landscape_state, params.steps)
        reports.water.block_until_ready()
        print(time.time() - t0)
        
        save_leaf_data(params, params_path)
        save_leaf_data(reports, reports_path)
    
