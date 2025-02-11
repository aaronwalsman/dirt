import time
import functools
from typing import Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.static_dataclass import static_dataclass
from mechagogue.dp.population_game import population_game

from dirt import heredity
from dirt.gridworld2d import dynamics, observations, spawn

TNomNomParams = TypeVar('TNomNomParams', bound='NomNomParams')
TNomNomState = TypeVar('TNomNomState', bound='NomNomState')
TNomNomObservation = TypeVar('TNomNomObservation', bound='NomNomObservation')
TNomNomAction = TypeVar('TNomNomAction', bound='NomNomAction')

@static_dataclass
class NomNomConfig:
    '''
    Configuration parameters.  This should only contain values that will be
    fixed throughout training.
    '''
    world_size : Tuple = (32,32)
    
    mean_initial_food : float = 32
    max_initial_food : float = 32
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

@static_dataclass
class NomNomState:
    '''
    State information about a single Nom environment.
    '''
    # grid-shaped data
    food_grid : jnp.ndarray
    object_grid : jnp.ndarray
    
    # player shaped data
    player_alive : jnp.ndarray
    player_x : jnp.ndarray
    player_r : jnp.ndarray
    player_energy : jnp.ndarray

@static_dataclass
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
        forward_key, rotate_key, reproduce_key = jrng.split(key, 3)
        forward = jrng.randint(forward_key, shape=(n,), minval=0, maxval=2)
        rotate = jrng.randint(rotate_key, shape=(n,), minval=-1, maxval=2)
        reproduce = jrng.randint(reproduce_key, shape=(n,), minval=0, maxval=2)
        return NomNomAction(forward, rotate, reproduce)

@static_dataclass
class NomNomObservation:
    '''
    An observation in the Nom environment.
    '''
    view : jnp.ndarray
    energy : jnp.ndarray

def nomnom(config):
    def init(
        key : chex.PRNGKey,
    ) -> TNomNomState :
        '''
        Returns a NomNomState object representing the start of a new episode.
        '''
        # initialize the players
        player_alive = jnp.zeros(config.max_players, dtype=jnp.bool)
        player_alive = alive.at[:config.initial_players].set(True)
        key, xr_key = jrng.split(key)
        player_x, player_r = spawn.unique_xr(
            xr_key, config.max_players, config.world_size)
        player_energy = jnp.full((config.max_players,), config.initial_energy)
        player_energy = player_energy * player_alive
        
        # initialize the object grid
        object_grid = jnp.full(config.world_size, -1, dtype=jnp.int32)
        object_grid.at[
            player_x[:config.initial_players,0],
            player_x[:config.initial_players,1]].set(
            jnp.arange(config.initial_players))
        
        # initialize the food grid
        key, food_key = jrng.split(key)
        food_grid = spawn.poisson_grid(
            food_key,
            config.mean_initial_food,
            config.max_initial_food,
            config.world_size,
        )
        
        # build the state
        state =  NomNomState(
            food_grid,
            object_grid,
            player_alive,
            player_x,
            player_r,
            player_energy,
        )

        return state

    def nomnom_transition(
        key: chex.PRNGKey,
        state: TNomNomState,
        action: TNomNomAction,
    ) -> TNomNomState :
        '''
        Transition function for the NomNom environment.  Samples a new state
        given a previous state and an action.
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
        # - figure out who will eat which food
        food_at_player = state.food_grid[player_x[...,0], player_x[...,1]]
        eaten_food = food_at_player * state.player_alive
        # - update the player energy with the food they have just eaten
        player_energy = jnp.clip(
            state.player_energy + eaten_food * config.food_metabolism,
            0,
            config.max_energy,
        )
        # - remove the eaten food from the food grid
        food_grid = state.food_grid.at[player_x[...,0], player_x[...,1]].set(
            food_at_player - eaten_food)
            #food_at_player & jnp.logical_not(eaten_food.astype(jnp.int32)))
        
        # metabolism
        moved = action.forward | (action.rotate != 0)
        player_energy = (
            player_energy +
            moved * config.move_metabolism +
            (1. - moved) * config.wait_metabolism
        ) * player_alive
        
        # kill players that have starved
        player_alive = player_alive & (player_energy > 0.)
        
        # update the object grid to account for dead players
        player_id = (
            jnp.arange(config.max_players) * player_alive + -1 * ~player_alive)
        object_grid = object_grid.at[player_x[...,0], player_x[...,1]].set(
            player_id)
        
        # make new players based on reproduction
        # - filter the reproduce vector to remove dead players and those without
        #   enough energy to create offspring
        reproduce = (
            action.reproduce &
            (player_energy > config.initial_energy) &
            player_alive
        )
        
        player_data, child_locations = heredity.produce_children(
            parents,
            (),
            birth_process,
        )
        
        # grow new food
        key, food_key = jrng.split(key)
        food_grid = food_grid | spawn.poisson_grid(
            food_key,
            config.mean_food_growth,
            config.max_food_growth,
            config.world_size,
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
        state: TNomNomState,
    ) -> TNomNomObservation :
        '''
        Computes the observation of a NomNom environment given the environment
        state.
        '''
        
        # construct a grid that contains class labels at each location
        # (0 = free space, 1 = food, 2 = player, 3 = out-of-bounds)
        view_grid = state.food_grid.astype(jnp.uint8)
        view_grid.at[state.player_x[...,0], state.player_x[...,1]].set(
            2 * (state.player_id != -1))
        
        # clip the viewing rectangles out for each player
        view = observations.first_person_view(
            state.player_x,
            state.player_r,
            view_grid,
            config.view_width,
            config.view_distance,
            out_of_bounds=3,
        )
        return NomNomObservation(view, state.player_energy)

    return population_game(
        init,
        transition,
        observe,
        alive,
        children,
    )
