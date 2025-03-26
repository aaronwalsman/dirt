from typing import Tuple, TypeVar

import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.static_dataclass import static_dataclass

from dirt.gridworld2d import dynamics, observations, spawn

TPootParams = TypeVar('TPootParams', bound='PootParams')
TPootState = TypeVar('TPootState', bound='PootState')
TPootObservation = TypeVar('TPootObservation', bound='PootObservation')
TPootAction = TypeVar('TPootAction', bound='PootAction')

'''
A gridworld full of poisonous plants.  You must poot (fart) on the plants to
make them edible.  However, you have to eat some poisonous plants to get more
poots.  The agent has two meters, health and poot juice.  Edible plants increase
health.  Poisonous plants reduce health, but increase poot juice.  Walking and
breathing costs health.  There is wind which blows the poots in random
directions.
'''

@static_dataclass
class PootParams:
    world_size : Tuple = (32,32)
    
    mean_initial_food : float = 32
    max_initial_food : float = 36
    mean_food_growth : float = 2
    max_food_growth : float = 4
    
    wind_acceleration : float = 0.05
    wind_drag : float = 0.05

    initial_health : float = 1.
    max_health : float = 1.
    move_metabolism : float = 0.05
    wait_metabolism : float = 0.025
    health_per_food : float = 0.5

    initial_poot_juice : float = 1.
    max_poot_juice : float = 1.
    poot_juice_per_food : float = 0.5
    poot_size : float = 0.25
    poot_memory_size : int = 8
    poot_filter_size : int = 3
    poot_diffusion : float = 0.5

    view_width : int = 7
    view_distance : int = 7
    
@static_dataclass
class PootState:
    food_grid : jnp.ndarray
    poot_grid : jnp.ndarray
    wind : jnp.ndarray
    agent_x : jnp.ndarray
    agent_r : jnp.ndarray
    agent_health : jnp.ndarray
    agent_poot_juice : jnp.ndarray
    agent_poot_history = jnp.ndarray

@static_dataclass
class PootAction:
    forward : bool
    rotate : int
    poot : bool
    
    @classmethod
    def sample(cls, key, state):
        key, forward_key, rotate_key, poot_key = jrng.split(key, 4)
        forward = jrng.randint(forward_key, shape=(1,), min_val=0, maxval=2)
        rotate = jrng.randint(rotate_key, shape=(1,), minval=-1, maxval=2)
        poot = jrng.randint(poot_key, shape=(1,), min_val=0, maxval=2)
        return PootAction(forward, rotate)

@static_dataclass
class PootObservation:
    view : jnp.ndarray
    wind : jnp.ndarray
    health : jnp.ndarray
    poot_juice : jnp.ndarray
    poot_history : jnpt.ndarray
    
    @classmethod
    def zero(cls, params: TPootParams):
        return PootObservation(
            view=jnp.zeros(
                (params.view_distance, params.view_width),
                dtype=jnp.int32,
            ),
            wind=jnp.zeros(2, dtype=jnp.float32)
            health=jnp.zeros(1, dtype=jnp.float32)
            poot_juice=jnp.zeros(1, dtype=jnp.float32)
        )

def observe(
    params: TPootParams,
    state: TPootState,
) -> TPootObservation :
    '''
    Computes the observation of a Poot environment given the environment params
    and state.
    
    When used inside a jit compiled program, params must come from a static
    variable as it controls the shapes of various arrays.
    '''
    view = observations.first_person_view(
        state.agent_x,
        state.agent_r,
        state.food_grid.astype(jnp.uint8),
        params.view_width,
        params.view_distance,
        out_of_bounds=2,
    )
    return PootObservation(
        view,
        state.wind,
        state.agent_health,
        state.agent_poot_juice,
    )

def reset(
    key: chex.PRNGKey,
    params: TPootParams,
) -> Tuple[TPootObservation, TPootState] :
    '''
    Reset function for the Poot environment.  Returns a Poot environment
    observation and state representing the start of a new episode.
    
    When used inside a jit compiled program, params must come from a static
    variable as it controls the shapes of various arrays.
    '''
    # initialize the agent
    key, xr_key = jrng.split(key)
    agent_x, agent_r = spawn.unique_xr(
        xr_key, 1, params.world_size)
    agent_x = agent_x
    agent_r = agent_r
    agent_health = jnp.full((1,), params.initial_health)
    agent_poot_juice = jnp.full((1,), params.initial_poot_juice)
    agent_poot_history = jnp.zeros((params.poot_memory_size,), dtype=jnp.uint8)

    # initialize the food grid
    key, foodkey = jrng.split(key)
    food_grid = spawn.poisson_grid(
        foodkey,
        params.mean_initial_food,
        params.max_initial_food,
        params.world_size,
    )
    
    # initialize the poot grid
    poot_grid = jnp.zeros(params.world_size, dtype=jnp.float32)
    
    # initialize the wind
    wind = jnp.zeros(2, dtype=jnp.float32)
    
    state =  PootState(
        food_grid,
        poot_grid,
        wind,
        agent_x,
        agent_r,
        agent_health,
        agent_poot_juice,
        agent_poot_history,
    )

    obs = observe(params, state)

    return obs, state

def step(
    key: chex.PRNGKey,
    params: TPootParams,
    state: TPootState,
    action: TPootAction,
) -> Tuple[TPootObservation, TPootState, jnp.ndarray, jnp.ndarray] :
    '''
    Transition function for the Poot environment.  Returns a new observation,
    state, reward and done given the environment params, a previous state
    and an action.
    '''

    # move
    agent_x, agent_r = dynamics.move_forward_turn(
        state.agent_x,
        state.agent_r,
        action.forward,
        action.rotate,
        params.world_size,
    )

    # eat
    eaten_food = food_grid[agent_x[0], agent_x[1]] * params.health_per_food
    agent_health = state.agent_health + eaten_food

    # grow food
    key, food_key = jrng.split(key)
    food_grid = spawn.poisson_grid(
        food_key,
        params.mean_food_growth,
        params.max_food_growth,
        params.world_size,
    )
    food_grid = state.food_grid.at[food_grid].set(-1)

    # update wind
    key, wind_key = jrng.split(key)
    wind_force = jrng.normal(wind_key, shape=(2,)) * params.wind_acceleration
    new_wind = (state.wind * (1 - params.wind_drag) + wind_force)

    # shift poot grid based on wind
    wind_direction = jnp.clip(new_wind, -1, 1)
    shifted_poot = dynamics.shift_grid(state.poot_grid, wind_direction)

    # apply gaussian blur to diffuse poot
    kernel_size = params.poot_filter_size
    sigma = params.poot_diffusion
    x = jnp.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    gaussian_kernel = jnp.exp(-x**2 / (2 * sigma**2))
    gaussian_kernel = gaussian_kernel[:, None] * gaussian_kernel[None, :]
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    # pad the poot grid for convolution
    padded_poot = jnp.pad(shifted_poot, kernel_size//2, mode='wrap')
    
    # apply convolution
    new_poot_grid = jax.lax.conv(
        padded_poot[None, None, :, :],
        gaussian_kernel[None, None, :, :],
        (1, 1),
        'VALID'
    )[0, 0]

    # add new poots if action.poot is True
    poot_mask = dynamics.circular_mask(
        new_x, params.world_size, params.poot_size)
    new_poot_grid = jnp.where(
        action.poot[:, None, None] & (state.agent_poot_juice >= 0.1),
        new_poot_grid + poot_mask,
        new_poot_grid
    )
    new_poot_grid = jnp.clip(new_poot_grid, 0, 1)

    # update poot juice
    new_poot_juice = jnp.where(
        action.poot & (state.agent_poot_juice >= 0.1),
        state.agent_poot_juice - 0.1,
        state.agent_poot_juice
    )

    # update food grid and collect rewards
    food_eaten = dynamics.collect_points(state.food_grid, new_x)
    
    # calculate health changes
    poisoned_food = food_eaten & (new_poot_grid[new_x[:, 0], new_x[:, 1]] < 0.5)
    safe_food = food_eaten & (new_poot_grid[new_x[:, 0], new_x[:, 1]] >= 0.5)
    
    health_delta = (safe_food * 0.1 -  # bonus for eating safe food
                   poisoned_food * 0.2 -  # penalty for eating poisoned food
                   metabolism)  # basic metabolism cost
    
    new_health = jnp.clip(
        state.agent_health + health_delta,
        0,
        params.max_health
    )

    # get poot juice from poisoned food
    new_poot_juice = jnp.clip(
        new_poot_juice + poisoned_food * params.poot_juice_per_food,
        0,
        params.max_poot_juice
    )

    # update poot history
    new_poot_history = jnp.roll(state.agent_poot_history, 1)
    new_poot_history = new_poot_history.at[0].set(action.poot[0])

    # create new state
    new_state = PootState(
        new_food_grid,
        new_poot_grid,
        new_wind,
        new_x,
        new_r,
        new_health,
        new_poot_juice,
        new_poot_history,
    )

    # get observation of new state
    obs = observe(params, new_state)

    # calculate reward and done
    reward = safe_food * 1.0  # reward for eating safe food
    done = new_health <= 0

    return obs, new_state, reward, done
