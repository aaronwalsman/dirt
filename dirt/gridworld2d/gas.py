import jax
import jax.numpy as jnp

from dirt.constants import DEFAULT_FLOAT_DTYPE

def gas(
    diffusion_std,
    diffusion_mix,
    boundary='clip',
    clip_fill=0.,
    float_dtype=DEFAULT_FLOAT_DTYPE,
):
    
    kernel_radius = jnp.ceil(3 * diffusion_std).astype(int)
    x = jnp.arange(-kernel_radius, kernel_radius + 1)
    kernel = jnp.exp(-x**2 / (2 * diffusion_std**2))
    kernel = kernel / kernel.sum()
    kernel = kernel[:, None]
    kernel = kernel[..., None, None]
    kernel = kernel.astype(float_dtype)
    
    def diffuse_step(grid):
        if len(grid.shape) == 2:
            grid = grid[:,:,None]
            remove_last_channel = True
        else:
            remove_last_channel = False
        assert len(grid.shape) == 3
        h,w,c = grid.shape
        diffused_grid = grid[None,:,:,:]
        
        channel_kernel = jnp.tile(kernel, (1, 1, 1, c))
        diffused_grid = jax.lax.conv_general_dilated(
            diffused_grid,
            channel_kernel,
            window_strides=(1,1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=c,
        )
        diffused_grid = jax.lax.conv_general_dilated(
            diffused_grid,
            channel_kernel.transpose((1,0,2,3)),
            window_strides=(1,1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=c,
        )
        diffused_grid = diffused_grid[0,:,:,:]
        
        grid = grid * (1 - diffusion_mix) + diffused_grid * diffusion_mix
        
        if remove_last_channel:
            grid = grid[...,0]
        
        return grid

    def wind_step(grid, wind):
        grid = jnp.roll(grid, shift=wind, axis=(0,1))
        if boundary == 'clip':
            h,w = grid.shape[:2]
            wind_y, wind_x = wind
            grid = grid.at[0:wind_y].set(clip_fill)
            grid = grid.at[h+wind_y:h].set(clip_fill)
            grid = grid.at[:,0:wind_x].set(clip_fill)
            grid = grid.at[:,w+wind_x:w].set(clip_fill)
        
        return grid
    
    def step(grid, wind):
        grid = wind_step(grid, wind)
        grid = diffuse_step(grid)
        return grid
    
    return step

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
