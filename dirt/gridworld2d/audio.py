import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve

def create_inverse_square_kernel(kernel_size: int, epsilon: float = 1e-3):
    """Creates a normalized inverse-square kernel centered at (c, c)."""
    c = kernel_size // 2
    x = jnp.arange(kernel_size) - c
    y = jnp.arange(kernel_size) - c
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    d2 = xx**2 + yy**2
    kernel = 1.0 / (d2 + epsilon)
    kernel = kernel / jnp.sum(kernel)
    return kernel

def inject_sources_multi_channel(grid_shape, num_channels, locations, signals):
    """
    Inject multi-channel signals at given locations into a (H, W, C) grid.
    """
    field = jnp.zeros((*grid_shape, num_channels))
    for loc, signal in zip(locations, signals):
        i, j = loc
        field = field.at[i, j, :].add(signal)
    return field

def sample_audio(
    source_schedule,            # List of (locations, signals) tuples per timestep
    sampled_locations,          # List of (i, j) tuples
    grid_shape=(64, 64),
    kernel_size=9,
    decay=0.95,
    num_steps=10
):
    """
    Simulate multi-channel sound propagation with convolution and dynamic source injection.
    
    Each timestep, source_schedule[t] = (locations_t, signals_t)
    signals_t should be of shape (num_sources_t, num_channels)
    """
    # Infer number of channels from the first non-empty signals
    for _, signals in source_schedule:
        if len(signals) > 0:
            num_channels = signals[0].shape[-1]
            break
    else:
        raise ValueError("No non-empty signal in schedule to infer channel count.")

    # Initialize field
    field = jnp.zeros((*grid_shape, num_channels))

    # Kernel for diffusion
    kernel = create_inverse_square_kernel(kernel_size)

    for t in range(num_steps):
        # Step 1: Apply per-channel convolution
        for c in range(num_channels):
            field = field.at[..., c].set(
                decay * convolve(field[..., c], kernel, mode='same')
            )
        
        # Step 2: Inject source for this timestep
        if t < len(source_schedule):
            locations_t, signals_t = source_schedule[t]
            source_field = inject_sources_multi_channel(grid_shape, num_channels, locations_t, signals_t)
            field += source_field

    # Sample the final field at desired locations
    sampled_signals = jnp.array([field[i, j, :] for (i, j) in sampled_locations])
    return sampled_signals

if __name__ == '__main__':
    grid_shape = (64, 64)
    num_channels = 2
    
    # Define a dynamic source schedule over 5 timesteps
    source_schedule = [
        ([(32, 32)], [jnp.array([1.0, 0.5])]),
        ([(33, 33)], [jnp.array([0.8, 0.4])]),
        ([(34, 34)], [jnp.array([0.6, 0.6])]),
        ([], []), 
        ([(30, 35)], [jnp.array([1.2, 0.2])]),
    ]
    
    sampled_locations = [(30, 30), (32, 32), (34, 34), (36, 36)]

    result = sample_audio(
        source_schedule,
        sampled_locations,
        grid_shape=grid_shape,
        kernel_size=7,
        decay=0.1,
        num_steps=10
    )

    print("Sampled multi-channel signals:\n", result)