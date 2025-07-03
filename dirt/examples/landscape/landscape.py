import time
from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.commandline import commandline_interface
from mechagogue.static import static_functions, static_data
from mechagogue.serial import save_leaf_data

from dirt.constants import DEFAULT_FLOAT_DTYPE
from dirt.visualization.viewer import Viewer
from dirt.gridworld2d.landscape import LandscapeParams, make_landscape
from dirt.gridworld2d.grid import grid_sum_to_mean

@commandline_interface()
@static_data
class LandscapeExampleParams:
    seed : int = 1234
    
    output_directory : str = '.'
    visualize : bool = False
    window_size : Tuple[int, int] = (512,512)
    
    landscape_params : LandscapeParams = LandscapeParams()
    
    steps : int = 1000

@static_data
class LandscapeExampleReport:
    rock : jnp.ndarray = False
    water : jnp.ndarray = False
    temperature : jnp.ndarray = False
    energy : jnp.ndarray = False
    biomass : jnp.ndarray = False
    moisture : jnp.ndarray = False
    raining : jnp.ndarray = False
    light : jnp.ndarray = False

def make_report(state):
    report = LandscapeExampleReport(
        rock=state.rock,
        water=state.water,
        temperature=state.temperature,
        energy=state.energy,
        biomass=state.biomass,
        moisture=state.moisture,
        raining=state.raining,
        light=state.light,
    )
    
    return report

if __name__ == '__main__':
    
    float_dtype = DEFAULT_FLOAT_DTYPE
    
    params = LandscapeExampleParams().from_commandline()
    landscape_params = params.landscape_params.replace(
        include_smell=False,
        include_audio=False,
    )
    params = params.replace(landscape_params = landscape_params)
    
    key = jrng.key(params.seed)
    
    landscape = make_landscape(landscape_params, float_dtype=float_dtype)
    
    params_path = f'{params.output_directory}/params.state'
    reports_path = f'{params.output_directory}/report.state'
    
    if params.visualize:
        key, init_key = jrng.split(key)
        
        def get_texture(report, texture_size, display_mode):
            
            if display_mode == 1:
                texture = landscape.render_rgb(report, shape=texture_size)
            
            elif display_mode == 2:
                texture = landscape.render_rgb(
                    report, shape=texture_size, use_light=False)
            
            elif display_mode == 3:
                texture = landscape.render_temperature(
                    report, shape=texture_size)
            
            elif display_mode == 4:
                texture = landscape.render_weather(
                    report, shape=texture_size)
            
            elif display_mode == 5:
                texture = landscape.render_altitude(
                    report, shape=texture_size)
            
            else:
                texture = jnp.zeros((*texture_size, 3))
            
            texture = np.array(
                jnp.clip(texture, min=0., max=1.) * 255).astype(np.uint8)
            return texture
        
        def get_terrain(report):
            return grid_sum_to_mean(
                report.rock + report.water,
                params.landscape_params.terrain_downsample,
            )
        
        key, init_key = jrng.split(key)
        example_state = landscape.init(init_key)
        example_report = make_report(example_state)
        
        viewer = Viewer(
            LandscapeExampleParams(),
            params_path,
            example_report,
            [reports_path],
            params.landscape_params.world_size,
            window_size=params.window_size,
            get_terrain_map = get_terrain,
            get_active_players = None,
            get_terrain_texture = get_texture,
            get_water_map = None,
            get_sun_direction = None,
        )
        viewer.begin()
    
    else:
        
        key, init_key = jrng.split(key)
        landscape_state = landscape.init(init_key)
        orig_landscape_state = landscape_state
        
        def landscape_step(key_state, _):
            key, state = key_state
            key, step_key = jrng.split(key)
            
            state = landscape.step(step_key, state)
            report = make_report(state)
            
            #breakpoint()
            
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
        
        landscape_state = orig_landscape_state
        
        t0 = time.time()
        (key, landscape_state), reports = run_scan(
            key, landscape_state, params.steps)
        reports.water.block_until_ready()
        print(time.time() - t0)
        
        save_leaf_data(params, params_path)
        save_leaf_data(reports, reports_path)
    
