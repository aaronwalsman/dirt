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
    diffusion_std=0.333,
    diffusion_radius=1,
    dissipation=0.,
    diffusion_type='box',
    boundary='clip',
    clip_fill=0.,
    include_diffusion=True,
    include_wind=True,
    max_wind=16,
    float_dtype=DEFAULT_FLOAT_DTYPE,
):
    
    assert world_size[0] % downsample == 0 and world_size[1] % downsample == 0
    assert boundary in ('clip', 'roll', 'collect', 'redistribute')
    downsample_size = (world_size[0]//downsample, world_size[1]//downsample)
    grid_size = (*downsample_size, *cell_shape)
    
    max_wind_cells = int(math.ceil(max_wind / downsample))
    
    if include_wind:
        def wind_step(key, grid, wind):
            wind = jnp.clip(wind, -max_wind, max_wind)
            downsampled_wind = wind / downsample
            wind_lo = jnp.floor(downsampled_wind).astype(jnp.int32)
            wind_hi = jnp.ceil(downsampled_wind).astype(jnp.int32)
            rounding = jrng.bernoulli(key, downsampled_wind-wind_lo)
            discrete_wind = jnp.where(rounding, wind_hi, wind_lo)
            
            #if boundary != 'collect':
            grid = jnp.roll(grid, shift=discrete_wind, axis=(0,1))
            
            if boundary in ('clip', 'collect', 'redistribute'):
                
                xlo = jnp.arange(max_wind_cells)
                xlo_mask = (
                    (xlo >= discrete_wind[0]) &
                    (xlo < downsample_size[0] + discrete_wind[0])
                )
                xlo_mask = xlo_mask[:,None,*((None,)*len(cell_shape))]
                xhi = xlo + downsample_size[0] - max_wind_cells
                xhi_mask = (
                    (xhi >= discrete_wind[0]) &
                    (xhi < downsample_size[0] + discrete_wind[0])
                )
                xhi_mask = xhi_mask[:,None,*((None,)*len(cell_shape))]
                
                ylo = jnp.arange(max_wind_cells)
                ylo_mask = (
                    (ylo >= discrete_wind[1]) &
                    (ylo < downsample_size[1] + discrete_wind[1])
                )
                ylo_mask = ylo_mask[None,:,*((None,)*len(cell_shape))]
                yhi = ylo + downsample_size[1] - max_wind_cells
                yhi_mask = (
                    (yhi >= discrete_wind[1]) &
                    (yhi < downsample_size[1] + discrete_wind[1])
                )
                yhi_mask = yhi_mask[None,:,*((None,)*len(cell_shape))]
                
                if boundary == 'clip':
                    grid = grid.at[:max_wind_cells].multiply(xlo_mask)
                    grid = grid.at[-max_wind_cells:].multiply(xhi_mask)
                    grid = grid.at[:,:max_wind_cells].multiply(ylo_mask)
                    grid = grid.at[:,-max_wind_cells:].multiply(yhi_mask)
                
                if boundary == 'redistribute':
                    # get the gas that has blown over each edge
                    lo_lo_corner = grid[:max_wind_cells, :max_wind_cells]
                    lo_mid_edge = grid[
                        :max_wind_cells, max_wind_cells:-max_wind_cells]
                    lo_hi_corner = grid[:max_wind_cells, -max_wind_cells:]
                    mid_hi_edge = grid[
                        max_wind_cells:-max_wind_cells, -max_wind_cells:]
                    hi_hi_corner = grid[-max_wind_cells:, -max_wind_cells:]
                    hi_mid_edge = grid[
                        -max_wind_cells:, max_wind_cells:-max_wind_cells]
                    hi_lo_corner = grid[-max_wind_cells:, :max_wind_cells]
                    mid_lo_edge = grid[
                        max_wind_cells:-max_wind_cells, :max_wind_cells]
                    
                    # sum these chunks
                    redistribute_total = (
                        jnp.sum(lo_lo_corner * ~xlo_mask * ~ylo_mask) +
                        jnp.sum(lo_mid_edge * ~xlo_mask) +
                        jnp.sum(lo_hi_corner * ~xlo_mask * ~yhi_mask) +
                        jnp.sum(mid_hi_edge * ~yhi_mask) +
                        jnp.sum(hi_hi_corner * ~xhi_mask * ~yhi_mask) +
                        jnp.sum(hi_mid_edge * ~xhi_mask) +
                        jnp.sum(hi_lo_corner * ~xhi_mask * ~ylo_mask) +
                        jnp.sum(mid_lo_edge * ~ylo_mask)
                    )
                    
                    # zero out the blown-over content
                    grid = grid.at[:max_wind_cells].multiply(xlo_mask)
                    grid = grid.at[-max_wind_cells:].multiply(xhi_mask)
                    grid = grid.at[:,:max_wind_cells].multiply(ylo_mask)
                    grid = grid.at[:,-max_wind_cells:].multiply(yhi_mask)
                    
                    # redistribute the blown-over content to each grid cell
                    num_cells = (downsample_size[0]*downsample_size[1])
                    grid = grid + redistribute_total / num_cells
                
                elif boundary == 'collect_slow':
                    l0 = jnp.arange(downsample_size[0]) + discrete_wind[0]
                    l0 = jnp.clip(l0, 0, downsample_size[0]-1)
                    l1 = jnp.arange(downsample_size[1]) + discrete_wind[1]
                    l1 = jnp.clip(l1, 0, downsample_size[1]-1)
                    l0m, l1m = jnp.meshgrid(l0, l1, indexing='ij')
                    next_grid = jnp.zeros_like(grid)
                    next_grid = next_grid.at[
                        l0m.reshape(-1), l1m.reshape(-1)].add(grid.reshape(-1))
                    grid = next_grid
                
                elif boundary == 'collect':
                    lo_lo_corner = grid[:max_wind_cells, :max_wind_cells]
                    lo_mid_edge = grid[
                        :max_wind_cells, max_wind_cells:-max_wind_cells]
                    lo_hi_corner = grid[:max_wind_cells, -max_wind_cells:]
                    mid_hi_edge = grid[
                        max_wind_cells:-max_wind_cells, -max_wind_cells:]
                    hi_hi_corner = grid[-max_wind_cells:, -max_wind_cells:]
                    hi_mid_edge = grid[
                        -max_wind_cells:, max_wind_cells:-max_wind_cells]
                    hi_lo_corner = grid[-max_wind_cells:, :max_wind_cells]
                    mid_lo_edge = grid[
                        max_wind_cells:-max_wind_cells, :max_wind_cells]
                    
                    l0 = jnp.arange(downsample_size[0]) + discrete_wind[0]
                    l0 = jnp.clip(l0, 0, downsample_size[0]-1)
                    l0 = jnp.roll(l0, shift=discrete_wind[0])
                    l1 = jnp.arange(downsample_size[1]) + discrete_wind[1]
                    l1 = jnp.clip(l1, 0, downsample_size[1]-1)
                    l1 = jnp.roll(l1, shift=discrete_wind[1])
                    
                    def collect_block(
                        grid,
                        block,
                        start_x,
                        stop_x,
                        start_y,
                        stop_y,
                        mask,
                    ):
                        block0, block1 = jnp.meshgrid(
                            l0[start_x:stop_x],
                            l1[start_y:stop_y],
                            indexing='ij',
                        )
                        block0, block1 = block0.reshape(-1), block1.reshape(-1)
                        grid = grid.at[block0, block1].add(
                            (block * mask).reshape(-1))
                        return grid
                    
                    m = max_wind_cells
                    grid = collect_block(
                        grid,lo_lo_corner,0,m,0,m,(~xlo_mask|~ylo_mask))
                    grid = collect_block(
                        grid,lo_mid_edge,0,m,m,-m,~xlo_mask)
                    grid = collect_block(
                        grid,lo_hi_corner,0,m,-m,None,(~xlo_mask|~yhi_mask))
                    grid = collect_block(
                        grid,mid_hi_edge,m,-m,-m,None,~yhi_mask)
                    grid = collect_block(
                        grid,hi_hi_corner,-m,None,-m,None,(~xhi_mask|~yhi_mask))
                    grid = collect_block(
                        grid,hi_mid_edge,-m,None,m,-m,~xhi_mask)
                    grid = collect_block(
                        grid,hi_lo_corner,-m,None,0,m,(~xhi_mask|~ylo_mask))
                    grid = collect_block(
                        grid,mid_lo_edge,m,-m,0,m,~ylo_mask)
                    
                    '''
                    lolo0, lolo1 = jnp.meshgrid(
                        l0[:max_wind_cells],
                        l1[:max_wind_cells],
                        indexing='ij',
                    )
                    grid = grid.at[
                        lolo0.reshape(-1), lolo1.reshape(-1)].add(
                        lo_lo_corner.reshape(-1) *
                        (~xlo_mask | ~ylo_mask).reshape(-1))
                    
                    jax.debug.print('all {a}', a=jnp.all(grid == new_grid))
                    
                    lomid0, lomid1 = jnp.meshgrid(
                        l0[:max_wind_cells],
                        l1[max_wind_cells:-max_wind_cells],
                        indexing='ij',
                    )
                    grid = grid.at[
                        lomid0.reshape(-1), lomid1.reshape(-1)].add(
                        (lo_mid_edge * ~xlo_mask).reshape(-1))
                    
                    lohi0, lohi1 = jnp.meshgrid(
                        l0[:max_wind_cells],
                        l1[-max_wind_cells:],
                        indexing='ij',
                    )
                    grid = grid.at[
                        lohi0.reshape(-1), lohi1.reshape(-1)].add(
                        lo_hi_corner.reshape(-1) *
                        (~xlo_mask | ~yhi_mask).reshape(-1))
                    
                    midhi0, midhi1 = jnp.meshgrid(
                        l0[max_wind_cells:-max_wind_cells],
                        l1[-max_wind_cells:],
                        indexing='ij',
                    )
                    grid = grid.at[
                        midhi0.reshape(-1), midhi1.reshape(-1)].add(
                        (mid_hi_edge * ~yhi_mask).reshape(-1))
                    
                    hihi0, hihi1 = jnp.meshgrid(
                        l0[-max_wind_cells:],
                        l1[-max_wind_cells:],
                        indexing='ij',
                    )
                    grid = grid.at[
                        hihi0.reshape(-1), hihi1.reshape(-1)].add(
                        hi_hi_corner.reshape(-1) *
                        (~xhi_mask | ~yhi_mask).reshape(-1))
                    
                    himid0, himid1 = jnp.meshgrid(
                        l0[-max_wind_cells:],
                        l1[max_wind_cells:-max_wind_cells],
                        indexing='ij',
                    )
                    grid = grid.at[
                        himid0.reshape(-1), himid1.reshape(-1)].add(
                        (hi_mid_edge * ~xhi_mask).reshape(-1))
                    
                    hilo0, hilo1 = jnp.meshgrid(
                        l0[-max_wind_cells:],
                        l1[:max_wind_cells],
                        indexing='ij',
                    )
                    grid = grid.at[
                        hilo0.reshape(-1), hilo1.reshape(-1)].add(
                        hi_lo_corner.reshape(-1) *
                        (~xhi_mask | ~ylo_mask).reshape(-1))
                    
                    midlo0, midlo1 = jnp.meshgrid(
                        l0[max_wind_cells:-max_wind_cells],
                        l1[:max_wind_cells],
                        indexing='ij',
                    )
                    grid = grid.at[
                        midlo0.reshape(-1), midlo1.reshape(-1)].add(
                        (mid_lo_edge * ~ylo_mask).reshape(-1))
                    '''
                    
                    grid = grid.at[:max_wind_cells].multiply(xlo_mask)
                    grid = grid.at[-max_wind_cells:].multiply(xhi_mask)
                    grid = grid.at[:,:max_wind_cells].multiply(ylo_mask)
                    grid = grid.at[:,-max_wind_cells:].multiply(yhi_mask)
                
                elif False: #boundary == 'collect':
                    # get the gas that has blown over each edge
                    lo_lo_corner = grid[:max_wind_cells, :max_wind_cells]
                    lo_mid_edge = grid[
                        :max_wind_cells, max_wind_cells:-max_wind_cells]
                    lo_hi_corner = grid[:max_wind_cells, -max_wind_cells:]
                    mid_hi_edge = grid[
                        max_wind_cells:-max_wind_cells, -max_wind_cells:]
                    hi_hi_corner = grid[-max_wind_cells:, -max_wind_cells:]
                    hi_mid_edge = grid[
                        -max_wind_cells:, max_wind_cells:-max_wind_cells]
                    hi_lo_corner = grid[-max_wind_cells:, :max_wind_cells]
                    mid_lo_edge = grid[
                        max_wind_cells:-max_wind_cells, :max_wind_cells]
                    
                    grid = grid.at[:max_wind_cells].multiply(xlo_mask)
                    grid = grid.at[-max_wind_cells:].multiply(xhi_mask)
                    grid = grid.at[:,:max_wind_cells].multiply(ylo_mask)
                    grid = grid.at[:,-max_wind_cells:].multiply(yhi_mask)
                    
                    #l0 = jnp.arange(downsample_size[0])
                    #l0 = jnp.roll(l0, shift=discrete_wind[0]) + discrete_wind[0]
                    #l0 = jnp.clip(l0, 0, downsample_size[0]-1)
                    #l1 = jnp.arange(downsample_size[1])
                    #l1 = jnp.roll(l1, shift=discrete_wind[1]) + discrete_wind[1]
                    #l1 = jnp.clip(l1, 0, downsample_size[1]-1)
                    
                    #collect_location = jnp.where(
                    #    discrete_wind < 0, 0, downsample_size)
                    
                    l0_lo = jnp.arange(max_wind_cells)
                    l0_hi = jnp.arange(
                        downsample_size[0] - max_wind_cells, downsample_size[0])
                    l0 = jnp.concatenate((l0_lo, l0_hi))
                    l0 = jnp.roll(l0, shift=discrete_wind[0]) + discrete_wind[0]
                    l0 = jnp.clip(l0, 0, downsample_size[0]-1)
                    
                    l0_lo, l0_hi = l0[:max_wind_cells], l0[max_wind_cells:]
                    
                    l1_lo = jnp.arange(max_wind_cells)
                    l1_hi = jnp.arange(
                        downsample_size[1] - max_wind_cells, downsample_size[1])
                    l1 = jnp.concatenate((l1_lo, l1_hi))
                    l1 = jnp.roll(l1, shift=discrete_wind[1]) + discrete_wind[1]
                    l1 = jnp.clip(l1, 0, downsample_size[1]-1)
                    l1_lo, l1_hi = l1[:max_wind_cells], l1[max_wind_cells:]
                    
                    #lo_lo_locations = jnp.zeros(
                    #    (max_wind_cells, max_wind_cells, 2), dtype=jnp.int32)
                    #lo_lo_locations.at[...,0].set(l0_lo[:,None])
                    #lo_lo_locations.at[...,1].set(l1_lo[None,:])
                    
                    aa, bb = jnp.meshgrid(l0_lo, l1_lo)
                    
                    lo_lo_masked = (
                        lo_lo_corner * ~xlo_mask * ~ylo_mask)
                    jax.debug.print('lo_lo {lolo}', lolo=lo_lo_corner)
                    grid = grid.at[aa.reshape(-1), bb.reshape(-1)].add(
                        lo_lo_masked.reshape(-1))
                    
                    #jax.debug.print(
                    #    'lo {l}, {x}, {y}',
                    #    l=lo_lo_masked,
                    #    x=xlo_mask,
                    #    y=ylo_mask,
                    #)
                    
                    
                    #grid = jax.lax.scatter_add(
                    #    grid,
                    #    lo_lo_locations.reshape(-1,2),
                    #    lo_lo_corner.reshape(-1),
                    #    indices_are_sorted=False,
                    #    unique_indices=False,
                    #)
                    
                    '''
                    xa = jnp.arange(downsample_size[0])[:,None]
                    ya = jnp.arange(downsample_size[1])[None,:]
                
                    wrapped_x = (
                        ((discrete_wind[0] > 0) & (xa < discrete_wind[0])) |
                        ((discrete_wind[0] < 0) &
                            (xa >= discrete_wind[0] + downsample_size[0]))
                    )
                    wrapped_y = (
                        ((discrete_wind[1] > 0) & (ya < discrete_wind[1])) |
                        ((discrete_wind[1] < 0) &
                            (xa >= discrete_wind[1] + downsample_size[1]))
                    )
                    wrapped = wrapped_x | wrapped_y
                    wrapped_values = grid * wrapped
                    
                    grid_clean = grid * (~wrapped)
                    '''
                    
            
            jax.debug.print('sum {s}', s=jnp.sum(grid))
            return grid
    
    if include_diffusion:
        
        '''
        # make the kernel
        if diffusion_type == 'gaussian':
            radius = jnp.ceil(3 * iter_std).astype(int)
            x = jnp.arange(-radius, radius + 1)
            kernel = jnp.exp(-x**2 / (2 * iter_std**2)).astype(float_dtype)
            kernel = kernel / kernel.sum()
        elif diffusion_type == 'box':
            radius = jnp.ceil(
                (-1 + jnp.sqrt(1 + 12 * step_std ** 2)) / 2).astype(int)
            n = 2 * radius + 1
            kernel = jnp.ones(n, dtype=float_dtype) / n
        '''
        
        n = 2 * diffusion_radius + 1
        kernel = jnp.ones(n, dtype=float_dtype) / n
        
        kernel = kernel[:, None]
        kernel = kernel[..., None, None]
        pad = diffusion_radius #* diffusion_iterations
        
        def diffusion_step(grid):
            if len(grid.shape) == 2:
                grid = grid[:,:,None]
                remove_last_channel = True
            else:
                remove_last_channel = False
            assert len(grid.shape) == 3
            h,w,c = grid.shape
            diffused_grid = grid[None,:,:,:]
            
            diffused_grid = jnp.pad(
                diffused_grid,
                ((0,0), (pad,pad), (pad,pad), (0,0)),
                mode='edge',
            )
            
            channel_kernel = jnp.tile(kernel, (1, 1, 1, c))
            #for _ in range(diffusion_iterations):
            # vertical
            diffused_grid = jax.lax.conv_general_dilated(
                diffused_grid,
                channel_kernel,
                window_strides=(1,1),
                padding='VALID',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                feature_group_count=c,
            )
            
            # horizontal
            diffused_grid = jax.lax.conv_general_dilated(
                diffused_grid,
                channel_kernel.transpose((1,0,2,3)),
                window_strides=(1,1),
                padding='VALID',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                feature_group_count=c,
            )
            diffused_grid = diffused_grid[0]
            
            if remove_last_channel:
                diffused_grid = diffused_grid[...,0]
            
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

def step_old(
    gas_grid: jnp.ndarray,
    sigma: float,
    mix: float,
    wind: jnp.ndarray,
    C: int
) -> jnp.ndarray:
    '''
    Diffuses a gas grid based on a gaussian kernel using separable convolutions.
    '''

    # make the gaussian kernel
    kernel_radius = jnp.ceil(3 * sigma).astype(int)
    x = jnp.arange(-kernel_radius, kernel_radius + 1)
    kernel = jnp.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel[:, None]
    kernel = kernel[..., None, None]
    kernel = jnp.tile(kernel, (1, 1, 1, C))
    kernel = kernel.astype(gas_grid.dtype)

    # Reshape input for conv operations (needs 4D: NHWC format)
    #x = gas_grid.reshape(1, *gas_grid.shape, 1)
    # gas_grid = gas_grid[..., None] if gas_grid.ndim == 2 else gas_grid
    
    # Apply horizontal then vertical convolution
    diffused_gas_grid = gas_grid[None, :, :, :]
    diffused_gas_grid = jax.lax.conv_general_dilated(
        diffused_gas_grid,
        kernel,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=C
    )
    diffused_gas_grid = jax.lax.conv_general_dilated(
        diffused_gas_grid,
        kernel.transpose((1, 0, 2, 3)),
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=C
    )
    diffused_gas_grid = diffused_gas_grid[0,:,:,:]
    
    gas_grid = gas_grid * (1 - mix) + diffused_gas_grid * mix

    # apply wind with bilinear interpolation
    wind_y, wind_x = wind

    # Get the four nearest grid points
    h = jnp.arange(gas_grid.shape[0]) + wind_y # Change H/W according to the specification
    w = jnp.arange(gas_grid.shape[1]) + wind_x # Change H/W according to the specification
    wind_y, wind_x = jnp.meshgrid(h, w, indexing="ij")
    x0 = jnp.floor(wind_x).astype(int)
    y0 = jnp.floor(wind_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clip indices so they remain within grid bounds.
    y0 = jnp.clip(y0, 0, gas_grid.shape[0] - 1)
    x0 = jnp.clip(x0, 0, gas_grid.shape[1] - 1)
    y1 = jnp.clip(y1, 0, gas_grid.shape[0] - 1)
    x1 = jnp.clip(x1, 0, gas_grid.shape[1] - 1)
    
    # Calculate interpolation weights
    wx1 = wind_x - x0
    wy1 = wind_y - y0
    wx0 = 1 - wx1
    wy0 = 1 - wy1

    # # Ensure indices stay within bounds
    # x0 = jnp.clip(x0, 0, gas_grid.shape[1] - 1)
    # y0 = jnp.clip(y0, 0, gas_grid.shape[0] - 1)
    # x1 = jnp.clip(x1, 0, gas_grid.shape[1] - 1)
    # y1 = jnp.clip(y1, 0, gas_grid.shape[0] - 1)
    
    # Distribute gas to the four nearest points

    new_grid = jnp.zeros_like(gas_grid)

    new_grid = new_grid.at[y0, x0, :].add((gas_grid * ((wy0 * wx0)[..., None])).astype(gas_grid.dtype))
    new_grid = new_grid.at[y0, x1, :].add((gas_grid * ((wy0 * wx1)[..., None])).astype(gas_grid.dtype))
    new_grid = new_grid.at[y1, x0, :].add((gas_grid * ((wy1 * wx0)[..., None])).astype(gas_grid.dtype))
    new_grid = new_grid.at[y1, x1, :].add((gas_grid * ((wy1 * wx1)[..., None])).astype(gas_grid.dtype))

    return new_grid

if __name__ == '__main__':
    gas_grid = jnp.zeros((5, 7, 3))
    gas_grid = gas_grid.at[3, 3, :].set(1)
    result = step(gas_grid, 0.5, 0.5, (0.1, 0.5), 3)

    for c in range(result.shape[2]):
        print(f"Channel {c}:")
        print(result[:, :, c])
        print()
