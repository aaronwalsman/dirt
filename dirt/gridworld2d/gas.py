import math

import jax
import jax.numpy as jnp
import jax.random as jrng

from dirt.constants import DEFAULT_FLOAT_DTYPE

from mechagogue.static import static_data, static_functions

def make_gas(
    world_size,
    downsample=1,
    cell_shape=(),
    initial_value=0.,
    diffusion_radius=1,
    diffusion_strength=1.,
    dissipation=0.,
    boundary='clip',
    empty_value=0.,
    include_diffusion=True,
    include_wind=True,
    max_wind=16,
    float_dtype=DEFAULT_FLOAT_DTYPE,
):
    
    assert world_size[0] % downsample == 0 and world_size[1] % downsample == 0
    assert boundary in ('roll', 'edge', 'clip', 'collect', 'redistribute')
    downsample_size = (world_size[0]//downsample, world_size[1]//downsample)
    grid_size = (*downsample_size, *cell_shape)
    
    max_wind_cells = int(math.ceil(max_wind / downsample))
    
    if include_wind:
        def wind_step(key, grid, wind):
            # convert the real-valued wind to a discrete_wind offset at the
            # correct grid resolution
            wind = jnp.clip(wind, -max_wind, max_wind)
            downsampled_wind = wind / downsample
            wind_lo = jnp.floor(downsampled_wind).astype(jnp.int32)
            wind_hi = jnp.ceil(downsampled_wind).astype(jnp.int32)
            rounding = jrng.bernoulli(key, downsampled_wind-wind_lo)
            discrete_wind = jnp.where(rounding, wind_hi, wind_lo)
            
            # apply the discrete_wind offset
            if boundary == 'roll':
                grid = jnp.roll(grid, shift=discrete_wind, axis=(0,1))
            
            elif boundary == 'edge':
                m = max_wind_cells
                grid = jnp.pad(
                    grid,
                    pad_width=((m,m),(m,m),*(((0,0),)*len(cell_shape))),
                    mode='edge',
                )
                grid = jnp.roll(grid, shift=discrete_wind, axis=(0,1))
                grid = grid[m:-m,m:-m]
            
            elif boundary in ('clip', 'collect', 'redistribute'):
                grid = jnp.roll(grid, shift=discrete_wind, axis=(0,1))
                
                # compute the masks for gas blown over the edge
                ar_0 = jnp.arange(downsample_size[0])
                ar_1 = jnp.arange(downsample_size[1])
                mask_0 = (
                    (ar_0 >= discrete_wind[0]) &
                    (ar_0 < downsample_size[0] + discrete_wind[0])
                )
                mask_1 = (
                    (ar_1 >= discrete_wind[1]) &
                    (ar_1 < downsample_size[1] + discrete_wind[1])
                )
                cell_pad = (None,)*len(cell_shape)
                
                def fill_void(grid):
                    m = max_wind_cells
                    fill = 0
                    grid = grid.at[:m].set(jnp.where(
                        mask_0[:m,None,*cell_pad],
                        grid[:m],
                        fill,
                    ))
                    grid = grid.at[-m:].set(jnp.where(
                        mask_0[-m:,None,*cell_pad],
                        grid[-m:],
                        fill,
                    ))
                    grid = grid.at[:,:m].set(jnp.where(
                        mask_1[None,:m,*cell_pad],
                        grid[:,:m],
                        fill,
                    ))
                    grid = grid.at[:,-m:].set(jnp.where(
                        mask_1[None,-m:,*cell_pad],
                        grid[:,-m:],
                        fill,
                    ))
                    return grid
                
                if boundary == 'clip':
                    ## zero out any gas that was blown over the boundaries
                    #m = max_wind_cells
                    #grid = grid.at[:m].multiply(mask_0[:m,None,*cell_pad])
                    #grid = grid.at[-m:].multiply(mask_0[-m:,None,*cell_pad])
                    #grid = grid.at[:,:m].multiply(mask_1[None,:m,*cell_pad])
                    #grid = grid.at[:,-m:].multiply(mask_1[None,-m:,*cell_pad])
                    grid = fill_void(grid)
                
                if boundary == 'redistribute':
                    
                    def get_block_total(lo0, hi0, lo1, hi1):
                        block = grid[lo0:hi0,lo1:hi1]
                        mask = ~mask_0[lo0:hi0,None] | ~mask_1[None,lo1:hi1]
                        mask = mask[:,:,*cell_pad]
                        return jnp.sum(block * mask, axis=(0,1))
                    
                    m = max_wind_cells
                    redistribute_total = (
                        get_block_total(0,m,0,m) +
                        get_block_total(0,m,m,-m) +
                        get_block_total(0,m,-m,None) +
                        get_block_total(m,-m,-m,None) +
                        get_block_total(-m,None,-m,None) +
                        get_block_total(-m,None,m,-m) +
                        get_block_total(-m,None,0,m) +
                        get_block_total(m,-m,0,m)
                    )
                    
                    ## zero out any gas that was blown over the boundaries
                    #grid = grid.at[:m].multiply(mask_0[:m,None,*cell_pad])
                    #grid = grid.at[-m:].multiply(mask_0[-m:,None,*cell_pad])
                    #grid = grid.at[:,:m].multiply(mask_1[None,:m,*cell_pad])
                    #grid = grid.at[:,-m:].multiply(mask_1[None,-m:,*cell_pad])
                    grid = fill_void(grid)
                    
                    # redistribute the blown-over content to each grid cell
                    num_cells = (downsample_size[0]*downsample_size[1])
                    grid = grid + redistribute_total / num_cells
                
                elif boundary == 'collect':
                    
                    # compute the destination locations
                    dest_0 = ar_0 + discrete_wind[0]
                    dest_0 = jnp.clip(dest_0, 0, downsample_size[0]-1)
                    dest_0 = jnp.roll(dest_0, shift=discrete_wind[0])
                    dest_1 = ar_1 + discrete_wind[1]
                    dest_1 = jnp.clip(dest_1, 0, downsample_size[1]-1)
                    dest_1 = jnp.roll(dest_1, shift=discrete_wind[1])
                    
                    # define the function that will redistribute gas that blows
                    # over the border onto the correct edge
                    def collect_block(grid, lo0, hi0, lo1, hi1):
                        block = grid[lo0:hi0, lo1:hi1]
                        
                        block_dest_0 = dest_0[lo0:hi0]
                        block_dest_1 = dest_1[lo1:hi1]
                        coord_0, coord_1 = jnp.meshgrid(
                            block_dest_0, block_dest_1, indexing='ij')
                        coord_0 = coord_0.reshape(-1)
                        coord_1 = coord_1.reshape(-1)
                        
                        block_mask_0 = ~mask_0[lo0:hi0]
                        block_mask_1 = ~mask_1[lo1:hi1]
                        mask = block_mask_0[:,None] | block_mask_1[None,:]
                        
                        mask = mask[:,:,*cell_pad]
                        
                        grid = grid.at[coord_0, coord_1].add(
                            (block * mask).reshape(-1,*cell_shape))
                        return grid
                    
                    # apply the collect function to the 8 border regions
                    m = max_wind_cells
                    grid = collect_block(grid,0,m,0,m)
                    grid = collect_block(grid,0,m,m,-m)
                    grid = collect_block(grid,0,m,-m,None)
                    grid = collect_block(grid,m,-m,-m,None)
                    grid = collect_block(grid,-m,None,-m,None)
                    grid = collect_block(grid,-m,None,m,-m)
                    grid = collect_block(grid,-m,None,0,m)
                    grid = collect_block(grid,m,-m,0,m)
                    
                    ## zero out any gas that was blown over the boundaries
                    #grid = grid.at[:m].multiply(mask_0[:m,None,*cell_pad])
                    #grid = grid.at[-m:].multiply(mask_0[-m:,None,*cell_pad])
                    #grid = grid.at[:,:m].multiply(mask_1[None,:m,*cell_pad])
                    #grid = grid.at[:,-m:].multiply(mask_1[None,-m:,*cell_pad])
                    grid = fill_void(grid)
            
            return grid
    
    if include_diffusion:
        
        # build the box filter convolution
        n = 2 * diffusion_radius + 1
        center_kernel = jnp.zeros(
            n, dtype=float_dtype).at[diffusion_radius].set(1.)
        box_kernel = jnp.full(n, 1./n, dtype=float_dtype)
        kernel = (
            box_kernel * diffusion_strength +
            center_kernel * (1. - diffusion_strength)
        )
        
        def diffusion_step(grid):
            # reshape the grid into 1,h,w,c format for conv_general_dilated
            h,w,*reshape_channels = grid.shape
            grid = grid.reshape(h,w,-1)
            _,_,c = grid.shape
            diffused_grid = grid[None,:,:,:]
            
            # pad the grid
            r = diffusion_radius
            diffused_grid = jnp.pad(
                diffused_grid,
                ((0,0), (r,r), (r,r), (0,0)),
                mode='edge',
            )
            
            # vertical convolution
            vertical_kernel = jnp.tile(kernel[:,None,None,None], (1, 1, 1, c))
            diffused_grid = jax.lax.conv_general_dilated(
                diffused_grid,
                vertical_kernel,
                window_strides=(1,1),
                padding='VALID',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                feature_group_count=c,
            )
            
            # horizontal convolution
            horizontal_kernel = jnp.tile(kernel[None,:,None,None], (1, 1, 1, c))
            diffused_grid = jax.lax.conv_general_dilated(
                diffused_grid,
                horizontal_kernel,
                window_strides=(1,1),
                padding='VALID',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                feature_group_count=c,
            )
            
            # reshape to original size
            diffused_grid = diffused_grid[0]
            diffused_grid = diffused_grid.reshape(h,w,*reshape_channels)
            
            return diffused_grid
    
    @static_functions
    class Gas:
        
        def init():
            grid = jnp.zeros(grid_size, dtype=float_dtype)
            grid = grid.at[:,:].set(initial_value)
            return grid
        
        def step(key, grid, wind=None):
            if dissipation:
                grid = grid * (1. - dissipation)
            if include_wind:
                grid = wind_step(key, grid, wind)
            if include_diffusion:
                grid = diffusion_step(grid)
            return grid
        
        def read(grid, x):
            cell = x // downsample
            cell_value = grid[cell[...,0], cell[...,1]]
            return cell_value / downsample**2
        
        def write(grid, x, value):
            cell = x // downsample
            grid = grid.at[cell[...,0], cell[...,1]].set(value * downsample**2)
            return grid
        
        def add(grid, x, value):
            cell = x // downsample
            grid = grid.at[cell[...,0], cell[...,1]].add(value)
            return grid
    
    return Gas
