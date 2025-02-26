import os
import argparse

import numpy as np

import jax.numpy as jnp

from dirt.visualization.viewer import Viewer
from dirt.examples.nomnom.train_nomnom import NomNomTrainParams, NomNomReport

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

def start_viewer(output_directory):
    params_path = f'{output_directory}/train_params.state'
    report_paths = sorted([
        f'{output_directory}/{file_path}'
        for file_path in os.listdir(output_directory)
        if file_path.startswith('report') and file_path.endswith('.state')
    ])
    
    viewer = Viewer(
        NomNomTrainParams(),
        params_path,
        NomNomReport(),
        report_paths,
        get_terrain_texture=terrain_texture,
    )
    viewer.begin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_directory', type=str)
    args = parser.parse_args()
    start_viewer(args.output_directory)
