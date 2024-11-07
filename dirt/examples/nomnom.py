import time
import functools
from typing import Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrng

from flax.struct import dataclass

import chex

from dirt.gridworld2d import dynamics, observations, spawn

TNomNomParams = TypeVar('TNomNomParams', bound='NomNomParams')
TNomNomState = TypeVar('TNomNomState', bound='NomNomState')
TNomNomObservation = TypeVar('TNomNomObservation', bound='NomNomObservation')
TNomNomAction = TypeVar('TNomNomAction', bound='NomNomAction')

@dataclass
class NomNomParams:
    max_agents : int = 32
    initial_agents : int = 32
    
    world_size : Tuple = (32,32)
    
    mean_initial_food : float = 32
    max_initial_food : float = 36
    mean_food_growth : float = 2
    max_food_growth : float = 4
   
    initial_energy : float = 1.
    max_energy : float = 3.
    move_metabolism : float = 0.2
    wait_metabolism : float = 0.1
   
    initial_health : float = 1.
    bite_damage : float = 0.5
    step_heal : float = 0.01
    
    view_width : int = 5
    view_distance : int = 5

@dataclass
class NomNomState:
    '''
    State information about a single Nom environment.
    '''
    food_grid : jnp.ndarray
    #occupancy_grid : jnp.ndarray
    object_grid : jnp.ndarray
    agent_alive : jnp.ndarray
    agent_x : jnp.ndarray
    agent_r : jnp.ndarray
    agent_health : jnp.ndarray
    agent_energy : jnp.ndarray

@dataclass
class NomNomAction:
    '''
    An action in the Nom environment.
    '''
    forward : jnp.ndarray
    rotate : jnp.ndarray
    bite : jnp.ndarray

    @classmethod
    def sample(cls, key, state):
        n = state.agent_x.shape[0]
        key, forward_key, rotate_key, bite_key = jrng.split(key, 4)
        forward = jrng.randint(forward_key, shape=(n,), minval=0, maxval=2)
        rotate = jrng.randint(rotate_key, shape=(n,), minval=-1, maxval=2)
        bite = jrng.randint(bite_key, shape=(n,), minval=0, maxval=2)
        
        return NomNomAction(forward, rotate, bite)

@dataclass
class NomNomObservation:
    '''
    An observation in the Nom environment.
    '''
    view : jnp.ndarray
    health : jnp.ndarray
    energy : jnp.ndarray

def observe( 
    params: TNomNomParams,
    state: TNomNomState,
) -> TNomNomObservation :
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
    return NomNomObservation(view, state.agent_health, state.agent_energy)

def reset(
    key : chex.PRNGKey,
    params : TNomNomParams,
) -> Tuple[TNomNomObservation, TNomNomState] :
    '''
    Reset function for the NomNom environment.  Returns a NomNom environment
    observation and state representing the start of a new episode.
    
    When used inside a jit compiled program, params must come from a static
    variable as it controls the shapes of various arrays.
    '''
    # initialize the agents
    key, xr_key = jrng.split(key)
    agent_alive = jnp.zeros(params.max_agents, dtype=jnp.bool)
    agent_alive.at[params.initial_agents].set(True)
    agent_x, agent_r = spawn.unique_xr(
        xr_key, params.max_agents, params.world_size)
    agent_health = jnp.full((params.max_agents,), params.initial_health)
    agent_energy = jnp.full((params.max_agents,), params.initial_energy)
    
    # initialize the object grid
    object_grid = jnp.full(params.world_size, -1, dtype=jnp.int32)
    object_grid.at[agent_x[...,0], agent_x[...,1]].set(
        jnp.arange(params.max_agents))
    
    # initialize the food grid
    key, foodkey = jrng.split(key)
    food_grid = spawn.poisson_grid(
        foodkey,
        params.mean_initial_food,
        params.max_initial_food,
        params.world_size,
    )
    
    # build the state and observation
    state =  NomNomState(
        food_grid,
        object_grid,
        agent_alive,
        agent_x,
        agent_r,
        agent_health,
        agent_energy,
    )
    obs = observe(params, state)

    return obs, state

def step(
    key: chex.PRNGKey,
    params: TNomNomParams,
    state: TNomNomState,
    action: TNomNomAction,
) -> Tuple[TNomNomObservation, TNomNomState, jnp.ndarray, jnp.ndarray] :
    '''
    Transition function for the Nom Nom environment.  Returns a new observation,
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
    agent_x, agent_r, _, object_grid = dynamics.forward_rotate_step(
        state.agent_x,
        state.agent_r,
        action.forward,
        action.rotate,
        check_collisions=True,
        object_grid=state.object_grid,
    )

    # eat
    eaten_food = food_grid[agent_x[...,0], agent_x[...,1]]
    agent_energy = jnp.clip(
        state.agent_energy + eaten_food, 0, params.max_energy)
    food_grid = food_grid.at[agent_x[...,0], agent_x[...,1]].set(False)

    # digest
    moved = action.forward | action.rotate
    agent_energy = (
        agent_energy -
        moved * params.move_metabolism -
        ~moved * params.wait_metabolism
    ) * state.agent_alive
    
    # fight
    pass
    agent_health = state.agent_health
    
    # kill
    agent_alive = state.agent_alive & (agent_energy > 0.) & (agent_health > 0.)
   
    # compute new state
    state = NomNomState(
        food_grid,
        object_grid,
        agent_alive,
        agent_x,
        agent_r,
        agent_health,
        agent_energy,
    )

    # compute observation
    obs = observe(params, state)

    return obs, state

def test_run(key, params, steps):
    key, reset_key = jrng.split(key)
    obs, state = reset(reset_key, params)
    
    #for i in range(steps):
    def single_step(key_obs_state, _):
        key, obs, state = key_obs_state
        key, action_key, step_key = jrng.split(key, 3)
        action = NomNomAction.sample(action_key, state)
        obs, state = step(step_key, params, state, action)
        
        return (key, obs, state), None
    
    jax.lax.scan(single_step, (key, obs, state), length=steps)

if __name__ == '__main__':
    
    key = jrng.key(1234)
    params = NomNomParams()
    steps = 1000
    
    jit_test_run = jax.jit(test_run, static_argnums=(1,2))
    
    t0 = time.time()
    jit_test_run(key, params, steps)
    t1 = time.time()
    print(steps/(t1-t0), 'hz')
