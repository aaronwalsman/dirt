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
from dirt.wrappers import make_step_auto_reset

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
    agent_stomach : jnp.ndarray

@dataclass
class NomAction:
    '''
    An action in the Nom environment.
    '''
    forward : bool
    rotate : int
    
    @classmethod
    def sample(cls, key, state):
        key, forward_key = jrng.split(key)
        forward = jrng.randint(forward_key, shape=(1,), minval=0, maxval=2)
        key, rotate_key = jrng.split(key)
        rotate = jrng.randint(rotate_key, shape=(1,), minval=-1, maxval=2)
        return NomAction(forward, rotate)

@dataclass
class NomObservation:
    '''
    An observation in the Nom environment.
    '''
    view : jnp.ndarray
    stomach : jnp.ndarray

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
    agent_x, agent_r = spawn.unique_xr(
        xr_key, 1, params.world_size)
    agent_x = agent_x
    agent_r = agent_r
    agent_stomach = jnp.full((1,), params.initial_stomach)
    
    # initialize the food grid
    key, foodkey = jrng.split(key)
    food_grid = spawn.poisson_grid(
        foodkey,
        params.mean_initial_food,
        params.max_initial_food,
        params.world_size,
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
        params.mean_food_growth,
        params.max_food_growth,
        params.world_size,
    )
    
    # move
    agent_x, agent_r = dynamics.forward_rotate_step(
        state.agent_x,
        state.agent_r,
        action.forward,
        action.rotate,
    )
    
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

def test_run(
    key,
    params,
):
    key, reset_key = jrng.split(key)
    
    step_auto_reset = make_step_auto_reset(step, reset)
    
    #jit_reset = jax.jit(reset, static_argnums=(1,))
    #jit_step_auto_reset = jax.jit(step_auto_reset, static_argnums=(1,))
    obs, state = reset(reset_key, params)
    
    i = 0
    while True:
        t = time.time()
        key, action_key = jrng.split(key)
        action = NomAction.sample(action_key, state)
        
        key, step_key = jrng.split(key)
        obs, state, reward, done = step_auto_reset(
            step_key, params, state, action)
        i += 1
        t2 = time.time()
        if i % 100 == 0:
            print(1./(t2-t))
    

if __name__ == '__main__':
    
    key = jrng.key(1234)
    keys = jrng.split(key, 16)
    params = NomParams()
    
    # for some reason doing both outer and inner JIT speeds things up quite a
    # bit... why isn't just doing the out one good enough?  Consult the AI
    # probably.
    
    outer_jit = True
    if outer_jit:
        jax.jit(
            jax.vmap(test_run, in_axes=(0, None)),
            static_argnums=(1,),
        )(keys, params)
    else:
        jax.vmap(test_run, in_axes=(0, None))(keys, params)
