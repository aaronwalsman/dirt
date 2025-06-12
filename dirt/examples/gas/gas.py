import time

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.commandline import commandline_interface
from mechagogue.static import static_functions, static_data
from mechagogue.serial import save_leaf_data

from dirt.gridworld2d.gas import make_gas
from dirt.distribution.ou import make_ou_process
from dirt.constants import DEFAULT_FLOAT_DTYPE
from dirt.visualization.viewer import Viewer

@commandline_interface
@static_data
class GasParams:
    seed : int = 1234
    
    output_directory : str = '.'
    visualize : bool = False
    
    world_size = (1024,1024)
    downsample = 1
    steps = 1000
    
    wind_std = 3
    wind_reversion = 0.1
    wind_bias = (0,0)
    
    source_location = (100,100)
    source_rate = 1.
    
    boundary : str = 'clip'

if __name__ == '__main__':
    
    #float_dtype = DEFAULT_FLOAT_DTYPE
    float_dtype = jnp.float32
    
    params = GasParams().from_commandline()
    
    key = jrng.key(params.seed)
    
    wind = make_ou_process(
        params.wind_std * jnp.sqrt(2*params.wind_reversion),
        params.wind_reversion,
        jnp.array(params.wind_bias, dtype=float_dtype),
        dtype=float_dtype
    )
    
    gas = make_gas(
        world_size=params.world_size,
        downsample=params.downsample,
        boundary=params.boundary,
        cell_shape=(),
    )
        
    params_path = f'{params.output_directory}/params.state'
    reports_path = f'{params.output_directory}/report.state'
    
    if params.visualize:
        example_gas_state = gas.init()
        h,w = example_gas_state.shape[:2]
        
        def get_texture(report):
            report = report.reshape(h,w,-1)[:,:,0]
            texture = jnp.stack((report, report, report), axis=-1)
            texture = np.array(jnp.clip(texture, 0, 1) * 255).astype(np.uint8)
            return texture
        
        viewer = Viewer(
            GasParams(),
            params_path,
            example_gas_state,
            [reports_path],
            get_terrain_map = lambda : jnp.zeros((h,w), dtype=float_dtype),
            get_active_players = None,
            get_terrain_texture = get_texture,
            get_water_map = None,
            get_sun_direction = None,
        )
        viewer.begin()
    
    else:
        
        key, wind_key = jrng.split(key)
        wind_state = wind.init(wind_key)
        gas_state = gas.init()
        
        def gas_step(key_wind_gas, _):
            key, wind_state, gas_state = key_wind_gas
            key, wind_key, gas_key = jrng.split(key, 3)
            
            location = jnp.array(params.source_location, dtype=jnp.int32)
            gas_state = gas.add(gas_state, location, params.source_rate) 
            
            wind_state = wind.step(wind_key, wind_state)
            gas_state = gas.step(gas_key, gas_state, wind=wind_state)
            
            return (key, wind_state, gas_state), gas_state
        
        @jax.jit
        def run_scan(key, wind_state, gas_state, steps):
            key_wind_gas, gasses = jax.lax.scan(
                gas_step,
                (key, wind_state, gas_state),
                None,
                length=params.steps,
            )
            
            return key_wind_gas, gasses
        
        (key, wind_state, gas_state), g = run_scan(
            key, wind_state, gas_state, params.steps)
        g.block_until_ready()
        
        t0 = time.time()
        (key, wind_state, gas_state), gasses = run_scan(
            key, wind_state, gas_state, params.steps)
        gasses.block_until_ready()
        print(time.time() - t0)
        
        save_leaf_data(params, params_path)
        save_leaf_data(gasses, reports_path)
    
