from typing import Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrng

from flax.struct import dataclass

import chex

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
    
    initial_health : float = 1.
    max_health : float = 3.
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
    agent_health : jnp.ndarray

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
    health : jnp.ndarray
    
    @classmethod
    def zero(cls, params: TNomParams):
        return NomObservation(
            view=jnp.zeros(
                (params.view_distance, params.view_width),
                dtype=jnp.int32,
            ),
            health=jnp.zeros(1, dtype=jnp.float32),
        )

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
    return NomObservation(view, state.agent_health)

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
    agent_x = agent_x[0]
    agent_r = agent_r[0]
    agent_health = jnp.full((1,), params.initial_health)
    
    # initialize the food grid
    key, foodkey = jrng.split(key)
    food_grid = spawn.poisson_grid(
        foodkey,
        params.mean_initial_food,
        params.max_initial_food,
        params.world_size,
    )
    
    state =  NomState(food_grid, agent_x, agent_r, agent_health)
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
    
    # move
    agent_x, agent_r = dynamics.forward_rotate_step(
        state.agent_x,
        state.agent_r,
        action.forward,
        action.rotate,
        world_size=params.world_size,
    )
    
    # eat
    eaten_food = food_grid[agent_x[...,0], agent_x[...,1]]
    agent_health = jnp.clip(
        state.agent_health + eaten_food, 0, params.max_health)
    food_grid = food_grid.at[agent_x[...,0], agent_x[...,1]].set(False)
    
    # digest
    moved = action.forward | action.rotate
    agent_health = (
        agent_health -
        moved * params.move_metabolism -
        ~moved * params.wait_metabolism
    )

    # grow food
    key, food_key = jrng.split(key)
    food_grid = state.food_grid | spawn.poisson_grid(
        food_key,
        params.mean_food_growth,
        params.max_food_growth,
        params.world_size,
    )

    # compute new state
    state = NomState(food_grid, agent_x, agent_r, agent_health)
    
    # compute observation
    obs = observe(params, state)
    
    # compute reward and done
    done = agent_health <= 0
    reward = -(done.astype(jnp.float32))
    
    return obs, state, reward, done

def test_run(key, params, steps):
    key, reset_key = jrng.split(key)
    step_auto_reset = make_step_auto_reset(step, reset)
    obs, state = reset(reset_key, params)
    
    def single_step(key_obs_state, _):
        key, obs, state = key_obs_state
        key, action_key, step_key = jrng.split(key, 3)
        action = NomAction.sample(action_key, state)
        obs, state, reward, done = step_auto_reset(
            step_key, params, state, action)
        
        return (key, obs, state), None
    
    jax.lax.scan(single_step, (key, obs, state), length=steps)
    

if __name__ == '__main__':
    import time
    
    key = jrng.key(1234)
    keys = jrng.split(key, 16)
    params = NomParams()
    steps = 1000
    
    jit_test_run = jax.jit(test_run, static_argnums=(1,2))
    
    t0 = time.time()
    jit_test_run(key, params, steps)
    t1 = time.time()
    print(steps/(t1-t0), 'hz')
