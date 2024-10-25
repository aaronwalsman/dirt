import time
import functools
from typing import Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrng

from flax.struct import dataclass

import chex

from gymnax.environments.environment import Environment

from dirt.gridworld2d import dynamics, observations, spawn

TNomParams = TypeVar('TNomParams', bound='NomParams')
TNomState = TypeVar('TNomState', bound='NomState')
TNomObservation = TypeVar('TNomObservation', bound='NomObservation')
TNomAction = TypeVar('TNomAction', bound='NomAction')

@dataclass
class NomParams:
    '''
    Hyperparameters for the Nom environment that will remain constant for
    each episode.
    '''
    world_size : Tuple = (32,32)
    
    mean_initial_food : float = 32
    max_initial_food : float = 36
    mean_food_growth : float = 2
    max_food_growth : float = 4
    
    initial_stomach : float = 1.
    max_stomach : float = 3.
    move_metabolism : float = 0.2
    wait_metabolism : float = 0.1
    
    view_width : int = 5
    view_distance : int = 5

@dataclass
class NomState:
    '''
    State information about a single Nom environment.
    '''
    food_grid : jnp.ndarray
    agent_x : jnp.ndarray
    agent_r : jnp.ndarray
    agent_stomach : float

@dataclass
class NomAction:
    '''
    An action in the Nom environment.
    '''
    forward : bool
    rotate : int
    
    @classmethod
    def sample(cls, key):
        key, forward_key = jrng.split(key)
        forward = jrng.randint(forward_key, shape=(), minval=0, maxval=2)
        key, rotate_key = jrng.split(key)
        rotate = jrng.randint(rotate_key, shape=(), minval=-1, maxval=2)
        return NomAction(forward, rotate)

@dataclass
class NomObservation:
    '''
    An observation in the Nom environment.
    '''
    view : jnp.ndarray
    stomach : float

def observe(
    params: TNomParams,
    state: TNomState,
) -> TNomObservation :
    '''
    Computes the observation of a Nom environment given the environment params
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
    return NomObservation(view, state.agent_stomach)

def reset(
    key: chex.PRNGKey,
    params: TNomParams,
) -> Tuple[TNomObservation, TNomState] :
    '''
    Reset function for the Nom environment.  Returns a Nom environment
    observation and state representing the start of a new episode.
    
    When used inside a jit compiled program, params must come from a static
    variable as it controls the shapes of various arrays.
    '''
    # initialize the agent
    key, xr_key = jrng.split(key)
    agent_x, agent_r = spawn.uniform_xr(xr_key, params.world_size)
    agent_x = agent_x[0]
    agent_r = agent_r[0]
    agent_stomach = params.initial_stomach
    
    # initialize the food grid
    key, foodkey = jrng.split(key)
    food_grid = spawn.poisson_grid(
        foodkey,
        params.world_size,
        params.mean_initial_food,
        params.max_initial_food,
    )
    
    state =  NomState(food_grid, agent_x, agent_r, agent_stomach)
    obs = observe(params, state)
    
    return obs, state

def step(
    key: chex.PRNGKey,
    params: TNomParams,
    state: TNomState,
    action: TNomAction,
) -> Tuple[TNomObservation, TNomState, jnp.ndarray, jnp.ndarray] :
    '''
    Transition function for the Nom environment.  Returns a new observation,
    state, reward and done given the environment params, a previous state
    and an action.
    
    When used inside a jit compiled program, params must come from a static
    variable as it controls the shapes of various arrays.
    '''
    # grow new food
    key, food_key = jrng.split(key)
    food_grid = state.food_grid | spawn.poisson_grid(
        food_key,
        params.world_size,
        params.mean_food_growth,
        params.max_food_growth,
    )
    
    # move
    agent_x, agent_r = dynamics.forward_rotate_step(
        state.agent_x, state.agent_r, action.forward, action.rotate)
    
    # eat
    eaten_food = food_grid[agent_x[...,0], agent_x[...,1]]
    agent_stomach = jnp.clip(
        state.agent_stomach + eaten_food, 0, params.max_stomach)
    food_grid = food_grid.at[agent_x[...,0], agent_x[...,1]].set(False)
    
    # digest
    moved = action.forward | action.rotate
    agent_stomach = (
        agent_stomach -
        moved * params.move_metabolism -
        ~moved * params.wait_metabolism
    )
    
    # compute new state
    state = NomState(food_grid, agent_x, agent_r, agent_stomach)
    
    # compute observation
    obs = observe(params, state)
    
    # compute reward and done
    done = agent_stomach <= 0
    reward = -(done.astype(jnp.float32))
    
    return obs, state, reward, done

def step_auto_reset(
    key: chex.PRNGKey,
    state: TNomState,
    action: TNomAction,
    params: TNomParams,
) -> Tuple[TNomObservation, TNomState, jnp.ndarray, jnp.ndarray]:
    '''
    Transition function for the Nom environment with automatic resets.
    Returns a new observation, state, reward and done given the environment
    params, a previous state and an action.  If the episode is done, the
    environment is reset to begin a new episode and the new observation and
    state are returned.
    
    When used inside a jit compiled program, params must come from a static
    variable as it controls the shapes of various arrays.
    '''
    
    key, step_key = jax.random.split(key)
    obs_step, state_step, reward, done = step(step_key, params, state, action)
    key, reset_key = jax.random.split(key)
    obs_reset, state_reset = reset(reset_key, params)
    
    # Auto-reset environment based on termination
    state = jax.tree.map(
        lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
    )
    obs = jax.tree.map(
        lambda x, y: jax.lax.select(done, x, y), obs_reset, obs_step
    )
    return obs, state, reward, done

if __name__ == '__main__':
    
    key = jrng.key(1234)
    
    params = NomParams()
    
    key, reset_key = jrng.split(key)
    jit_reset = jax.jit(reset, static_argnums=(1,))
    obs, state = jit_reset(reset_key, params)
    
    i = 0
    while True:
        t = time.time()
        key, action_key = jrng.split(key)
        action = NomAction.sample(action_key)
        
        key, step_key = jrng.split(key)
        jit_step = jax.jit(step_auto_reset, static_argnums=(3,))
        obs, state, reward, done = jit_step(
            step_key, state, action, params)
        i += 1
        t2 = time.time()
        if i % 100 == 0:
            print(1./(t2-t))
