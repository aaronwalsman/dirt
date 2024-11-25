import jax
import jax.numpy as jnp

def step(
    gas_grid: jnp.ndarray,
    sigma: float,
    gas_diffusion: float,
    wind: jnp.ndarray,
) -> jnp.ndarray:
    '''
    Diffuses a gas grid based on a gaussian kernel using separable convolutions.
    '''

    # make the gaussian kernel
    kernel_radius = jnp.ceil(3 * sigma)
    x = jnp.arange(-kernel_radius, kernel_radius + 1)
    kernel = jnp.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Reshape input for conv operations (needs 4D: NHWC format)
    #x = gas_grid.reshape(1, *gas_grid.shape, 1)
    
    # Apply horizontal then vertical convolution
    diffused_gas_grid = gas_grid[None,:,:,None]
    diffused_gas_grid = jax.lax.conv_general_dilated(
        diffused_gas_grid,
        kernel[:,None,None,None],
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    diffused_gas_grid = jax.lax.conv_general_dilated(
        diffused_gas_grid,
        kernel[None,:,None,None],
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    diffused_gas_grid = diffused_gas_grid[0,:,:,0]
    
    gas_grid = (
        gas_grid * (1 - gas_diffusion) +
        diffused_gas_grid * gas_diffusion
    )

    # apply wind with bilinear interpolation
    wind_y, wind_x = wind

    # Get the four nearest grid points
    h = jnp.arange(gas_grid.shape[1]) + wind[1]
    w = jnp.arange(gas_grid.shape[0]) + wind[0]

    breakpoint()

    wind_x, wind_y = jnp.meshgrid(w, h)
    breakpoint()
    x0 = jnp.floor(wind_x).astype(int)
    y0 = jnp.floor(wind_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Calculate interpolation weights
    wx1 = wind_x - x0
    wy1 = wind_y - y0
    wx0 = 1 - wx1
    wy0 = 1 - wy1
    
    # Distribute gas to the four nearest points
    breakpoint()
    gas_grid = gas_grid.at[x0, y0].add(gas_grid * (wx0 * wy0))
    gas_grid = gas_grid.at[x1, y0].add(gas_grid * (wx1 * wy0))
    gas_grid = gas_grid.at[x0, y1].add(gas_grid * (wx0 * wy1))
    gas_grid = gas_grid.at[x1, y1].add(gas_grid * (wx1 * wy1))
    
    return gas_grid

if __name__ == '__main__':
    gas_grid = jnp.zeros((5, 7))
    gas_grid = gas_grid.at[3, 3].set(1)
    print(step(gas_grid, 0.5, 0.5, (0.1, 0.5)))
