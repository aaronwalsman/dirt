import jax
import jax.numpy as jnp

def step(
    gas_grid: jnp.ndarray,
    sigma: float,
    gas_diffusion: float,
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
    
    gas_grid = (
        gas_grid * (1 - gas_diffusion) +
        diffused_gas_grid * gas_diffusion
    )

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

    return gas_grid

if __name__ == '__main__':
    gas_grid = jnp.zeros((5, 7, 3))
    gas_grid = gas_grid.at[3, 3, :].set(1)
    result = step(gas_grid, 0.5, 0.5, (0.1, 0.5), 3)

    for c in range(result.shape[2]):
        print(f"Channel {c}:")
        print(result[:, :, c])
        print()
