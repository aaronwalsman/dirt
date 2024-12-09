import time
import functools
from typing import Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrng

from flax.struct import dataclass

import chex

from dirt.gridworld2d import dynamics, observations, spawn
from mechagogue.dp.population_game import population_game

TNomNomParams = TypeVar('TNomNomParams', bound='NomNomParams')
TNomNomState = TypeVar('TNomNomState', bound='NomNomState')
TNomNomObservation = TypeVar('TNomNomObservation', bound='NomNomObservation')
TNomNomAction = TypeVar('TNomNomAction', bound='NomNomAction')

@dataclass
class NomNomParams:
    world_size : Tuple = (32,32)
    
    mean_initial_food : float = 32
    max_initial_food : float = 36
    mean_food_growth : float = 2
    max_food_growth : float = 4
   
    initial_players : int = 32
    max_players : int = 256
    
    initial_energy : float = 1.
    max_energy : float = 5.
    food_metabolism : float = 1
    move_metabolism : float = -0.05
    wait_metabolism : float = -0.025
    
    view_width : int = 5
    view_distance : int = 5

@dataclass
class NomNomState:
    '''
    State information about a single Nom environment.
    '''
    food_grid : jnp.ndarray
    object_grid : jnp.ndarray
    
    player_id : jnp.ndarray
    parent_id : jnp.ndarray
    player_x : jnp.ndarray
    player_r : jnp.ndarray
    player_energy : jnp.ndarray
    
    next_new_player_id : int

@dataclass
class NomNomAction:
    '''
    An action in the Nom environment consists of three buttons:
    forward : [0,1] moves the agent forward one step
    rotate : [0,1,2] rotates the agent left, zero or right
    reproduce : [0,1] makes a new agent
    '''
    forward : jnp.ndarray
    rotate : jnp.ndarray
    reproduce : jnp.ndarray
    
    @classmethod
    def uniform_sample(cls, key, n):
        #n = state.player_x.shape[0]
        forward_key, rotate_key, reproduce_key = jrng.split(key, 3)
        forward = jrng.randint(forward_key, shape=(n,), minval=0, maxval=2)
        rotate = jrng.randint(rotate_key, shape=(n,), minval=-1, maxval=2)
        reproduce = jrng.randint(reproduce_key, shape=(n,), minval=0, maxval=2)
        #forward = jrng.randint(forward_key, minval=0, maxval=2)
        #rotate = jrng.randint(rotate_key, minval=-1, maxval=2)
        #reproduce = jrng.randint(reproduce_key, minval=0, maxval=2)
        
        return NomNomAction(forward, rotate, reproduce)

@dataclass
class NomNomObservation:
    '''
    An observation in the Nom environment.
    '''
    view : jnp.ndarray
    energy : jnp.ndarray

def nomnom_initialize(
    key : chex.PRNGKey,
    params : TNomNomParams,
) -> TNomNomState :
    '''
    Reset function for the NomNom environment.  Returns a NomNom environment
    observation and state representing the start of a new episode.
    
    When used inside a jit compiled program, params must come from a static
    variable as it controls the shapes of various arrays.
    '''
    # initialize the players
    key, xr_key = jrng.split(key)
    
    player_id = jnp.full(params.max_players, -1, dtype=jnp.int32)
    player_id = player_id.at[:params.initial_players].set(
        jnp.arange(params.initial_players))
    parent_id = jnp.full(params.max_players, -1, dtype=jnp.int32)
    player_x, player_r = spawn.unique_xr(
        xr_key, params.max_players, params.world_size)
    player_energy = jnp.full((params.max_players,), params.initial_energy)
    
    # initialize the object grid
    object_grid = jnp.full(params.world_size, -1, dtype=jnp.int32)
    object_grid.at[player_x[...,0], player_x[...,1]].set(player_id)
    
    # initialize the food grid
    key, foodkey = jrng.split(key)
    food_grid = spawn.poisson_grid(
        foodkey,
        params.mean_initial_food,
        params.max_initial_food,
        params.world_size,
    )
    
    # build the state
    state =  NomNomState(
        food_grid,
        object_grid,
        player_id,
        parent_id,
        player_x,
        player_r,
        player_energy,
        params.initial_players,
    )

    return state

def nomnom_transition(
    key: chex.PRNGKey,
    params: TNomNomParams,
    state: TNomNomState,
    action: TNomNomAction,
) -> TNomNomState :
    '''
    Transition function for the NomNom environment.  Samples a new state
    given the environment params, a previous state and an action.
    
    When used inside a jit compiled program, params must come from a static
    variable as it controls the shapes of various arrays.
    '''
    
    # move
    player_x, player_r, _, object_grid = dynamics.forward_rotate_step(
        state.player_x,
        state.player_r,
        action.forward,
        action.rotate,
        check_collisions=True,
        object_grid=state.object_grid,
    )

    # eat
    player_alive = (state.player_id != -1)
    food_at_player = state.food_grid[player_x[...,0], player_x[...,1]]
    eaten_food = food_at_player * player_alive
    player_energy = jnp.clip(
        state.player_energy + eaten_food * params.food_metabolism,
        0,
        params.max_energy,
    )
    
    food_grid = state.food_grid.at[player_x[...,0], player_x[...,1]].set(
        food_at_player & jnp.logical_not(eaten_food.astype(jnp.int32)))
    #jax.debug.print('pref: {pf}, eaten: {e}, postf: {post}',
    #    pf=jnp.sum(state.food_grid),
    #    e=jnp.sum(eaten_food),
    #    post=jnp.sum(food_grid),
    #)
    
    # metabolism
    moved = action.forward | (action.rotate != 0)
    player_energy = (
        player_energy +
        moved * params.move_metabolism +
        (1. - moved) * params.wait_metabolism
    ) * player_alive
    
    # kill players that have starved
    player_alive = player_alive & (player_energy > 0.)
    player_id = state.player_id * player_alive + -1 * ~player_alive
    
    # update the object grid with dead players
    object_grid = object_grid.at[player_x[...,0], player_x[...,1]].set(
        player_id)
    
    # make new players
    # in order to reproduce, a player must have enough energy and not be dead
    reproduce = (
        action.reproduce &
        (player_energy > params.initial_energy) &
        (player_id != -1)
    )
    
    (
        next_new_player_id,
        player_new,
        player_id,
        parent_id,
        player_x,
        player_r,
        player_energy,
        object_grid,
    ) = spawn.reproduce_from_parents(
        reproduce,
        state.next_new_player_id,
        player_id,
        state.parent_id,
        player_x,
        player_r,
        player_energy,
        jnp.full_like(player_energy, params.initial_energy),
        object_grid=object_grid,
    )
    
    #jax.debug.print('internal_p {p}', p=player_id)
    #jax.debug.print('energy {e}', e=player_energy)
    #jax.debug.print('eaten {e}', e=eaten_food)
    
    # grow new food
    key, food_key = jrng.split(key)
    food_grid = food_grid | spawn.poisson_grid(
        food_key,
        params.mean_food_growth,
        params.max_food_growth,
        params.world_size,
    )
    
    # compute new state
    state = NomNomState(
        food_grid,
        object_grid,
        player_id,
        parent_id,
        player_x,
        player_r,
        player_energy,
        next_new_player_id
    )
    
    return state

def nomnom_observe(
    key: chex.PRNGKey,
    params: TNomNomParams,
    state: TNomNomState,
) -> TNomNomObservation :
    '''
    Computes the observation of a NomNom environment given the environment
    params and state.
    
    When used inside a jit compiled program, params must come from a static
    variable as it controls the shapes of various arrays.
    '''
    
    view_grid = state.food_grid.astype(jnp.uint8)
    view_grid.at[state.player_x[...,0], state.player_x[...,1]].set(
        2 * (state.player_id != -1))
    
    view = observations.first_person_view(
        state.player_x,
        state.player_r,
        view_grid,
        params.view_width,
        params.view_distance,
        out_of_bounds=3,
    )
    return NomNomObservation(view, state.player_energy)

def nomnom(
    params: TNomNomParams = NomNomParams,
):
    reset, step = population_game(
        params,
        nomnom_initialize,
        nomnom_transition,
        nomnom_observe,
        lambda params, state : state.player_id,
        lambda params, state : state.parent_id,
    )
    
    return reset, step

def test_run(key, params, steps):
    
    reset, step = nomnom(params)
    
    key, reset_key = jrng.split(key)
    state, obs, players, parents = reset(reset_key)
    
    def single_step(step_state, _):
        key, state, obs, players, parents = step_state
        key, action_key, step_key = jrng.split(key, 3)
        
        action = NomNomAction.sample(action_key, state)
        
        state, obs, players, parents = step(step_key, params, state, action)
        
        return (key, state, players, parents), None
    
    jax.lax.scan(single_step, (key, state, obs, players, parents), length=steps)

if __name__ == '__main__':
    
    key = jrng.key(1234)
    params = NomNomParams()
    steps = 1000
    
    jit_test_run = jax.jit(test_run, static_argnums=(1,2))
    
    t0 = time.time()
    jit_test_run(key, params, steps)
    t1 = time.time()
    print(steps/(t1-t0), 'hz')
