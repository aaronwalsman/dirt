from typing import Tuple, TypeVar, Any

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.static_dataclass import static_dataclass
from mechagogue.dp.population_game import population_game
from mechagogue.player_list import birthday_player_list, player_family_tree

from dirt.gridworld2d import dynamics, observations, spawn

TNomNomParams = TypeVar('TNomNomParams', bound='NomNomParams')
TNomNomState = TypeVar('TNomNomState', bound='NomNomState')
TNomNomObservation = TypeVar('TNomNomObservation', bound='NomNomObservation')
TNomNomAction = TypeVar('TNomNomAction', bound='NomNomAction')

@static_dataclass
class NomNomParams:
    '''
    Configuration parameters.  This should only contain values that will be
    fixed throughout training.
    '''
    world_size : Tuple = (32,32)
    
    mean_initial_food : float = 32
    max_initial_food : int = 32
    mean_food_growth : float = 2
    max_food_growth : int = 4
   
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
    family_tree : Any
    player_x : jnp.ndarray
    player_r : jnp.ndarray
    player_energy : jnp.ndarray

    curr_step: jnp.int8

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

@static_dataclass
class NomNomObservation:
    '''
    An observation in the Nom environment.
    '''
    view : jnp.ndarray
    energy : jnp.ndarray

def nomnom(
    params: TNomNomParams = NomNomParams,
):
    '''
    This bundles the function above into reset and step functions.
    reset will take a random key and produce a new state, observation, list of
    players and their parents.
    step will take a random key, a previous state and an action and produce
    a new state, observation, list of players and their parents. 
    '''
    
    init_players, step_players, active_players = birthday_player_list(
        params.max_players)
    init_family_tree, step_family_tree, active_family_tree = player_family_tree(
        init_players, step_players, active_players, 1)
    
    def init_state(
        key : chex.PRNGKey,
    ) -> TNomNomState :
        '''
        Returns a NomNomState object representing the start of a new episode.
        '''
        # initialize the players
        family_tree = init_family_tree(params.initial_players)
        active_players = active_family_tree(family_tree)
        
        key, xr_key = jrng.split(key)
        player_x, player_r = spawn.unique_xr(
            xr_key,
            params.max_players,
            params.world_size,
            active=active_players,
        )
        
        #player_energy = jnp.full((params.max_players,), params.initial_energy)
        n_hot = jnp.arange(params.max_players) < params.initial_players
        player_energy = n_hot * params.initial_energy
        
        # initialize the object grid
        object_grid = dynamics.make_object_grid(
            params.world_size, player_x, active_players) 
    
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
            family_tree,
            player_x,
            player_r,
            player_energy,
            0
        )

        return state

    def observe(
        key: chex.PRNGKey,
        state: TNomNomState,
    ) -> TNomNomObservation :
        '''
        Computes the observation of a NomNom environment given the environment
        params and state.
        '''
    
        # construct a grid that contains class labels at each location
        # (0 = free space, 1 = food, 2 = player, 3 = out-of-bounds)
        view_grid = state.food_grid.astype(jnp.uint8)
        active_players = active_family_tree(state.family_tree)
        view_grid.at[state.player_x[...,0], state.player_x[...,1]].set(
            2 * active_players)
    
        # clip the viewing rectangles out for each player
        view = observations.first_person_view(
            state.player_x,
            state.player_r,
            view_grid,
            params.view_width,
            params.view_distance,
            out_of_bounds=3,
        )
        return NomNomObservation(view, state.player_energy)
    
    def transition(
        key: chex.PRNGKey,
        state: TNomNomState,
        action: TNomNomAction,
    ) -> TNomNomState :
        '''
        Transition function for the NomNom environment.  Samples a new state
        given the environment params, a previous state and an action.
        '''
        
        # TODO: It turns out that the order of operations in this code is
        # actually somewhat sensitive, especially considering the updates to
        # the object grid.  It turns out that there are many subtle bugs that
        # can come up if this is not all handled correctly.  I think an
        # it would be worthwhile in the near future to to take the components
        # here and find a way to compartmentalize them further and make a
        # computational primitive that handles the book-keeping of not only
        # the list of active players, but their positions, rotations and
        # the object_grid.
        
        # get the active players
        active_players = active_family_tree(state.family_tree)
        
        # move
        player_x, player_r, _, object_grid = dynamics.forward_rotate_step(
            state.player_x,
            state.player_r,
            action.forward,
            action.rotate,
            active=active_players,
            check_collisions=True,
            object_grid=state.object_grid,
        )
        
        # eat
        # - figure out who will eat which food
        food_at_player = state.food_grid[player_x[...,0], player_x[...,1]]
        eaten_food = food_at_player * active_players
        # - update the player energy with the food they have just eaten
        player_energy = jnp.clip(
            state.player_energy + eaten_food * params.food_metabolism,
            0,
            params.max_energy,
        )
        # - remove the eaten food from the food grid
        food_grid = state.food_grid.at[player_x[...,0], player_x[...,1]].set(
            food_at_player & jnp.logical_not(eaten_food.astype(jnp.int32)))
    
        # metabolism
        moved = action.forward | (action.rotate != 0)
        player_energy = (
            player_energy +
            moved * params.move_metabolism +
            (1. - moved) * params.wait_metabolism
        ) * active_players

        # update the players based on starvation and reproduction
        # - filter the reproduce vector to remove dead players and those without
        #   enough energy to create offspring
        reproduce = (
            action.reproduce &
            (player_energy > params.initial_energy) &
            active_players
        )
        
        # - generate the new child locations (filter out colliding children)
        reproduce, child_x, child_r = spawn.spawn_from_parents(
            reproduce,
            player_x,
            player_r,
            object_grid=object_grid,
        )
    
        # - kill players that have starved
        deaths = player_energy <= 0.
        
        # - use reproduce to make new child ids and update the player data
        n = reproduce.shape[0]
        parent_locations, = jnp.nonzero(reproduce, size=n, fill_value=n)
        parent_locations = parent_locations[...,None]
        family_tree, child_locations = step_family_tree(
            state.family_tree, deaths, parent_locations)
        
        # - reorder child_x and child_r to be aligned with the parent
        #   and child locations
        child_x = child_x[parent_locations[...,0]]
        child_r = child_r[parent_locations[...,0]]
        
        # update the object grid, player positions and rotations based on deaths
        # - update the object grid before the positions
        object_grid = object_grid.at[player_x[...,0], player_x[...,1]].set(
            jnp.where(deaths, -1, jnp.arange(n)))
        # - then update the positions and rotations
        player_x = jnp.where(
            deaths[:,None],
            jnp.array(params.world_size, dtype=jnp.int32),
            player_x,
        )
        player_r = jnp.where(deaths, 0, player_r)
        
        # update the object grid, player positions, rotation and energy based on
        # new children
        object_grid = object_grid.at[child_x[...,0], child_x[...,1]].set(
            child_locations)
        player_x = player_x.at[child_locations].set(child_x)
        player_r = player_r.at[child_locations].set(child_r)
        player_energy = player_energy.at[parent_locations].add(
            -params.initial_energy)
        player_energy = player_energy.at[child_locations].set(
            params.initial_energy)
        
        # grow new food
        key, food_key = jrng.split(key)
        new_food = spawn.poisson_grid(
            food_key,
            params.mean_food_growth,
            params.max_food_growth,
            params.world_size,
        )
        food_grid = food_grid | new_food

        # compute new state
        state = NomNomState(
            food_grid,
            object_grid,
            family_tree,
            player_x,
            player_r,
            player_energy,
            state.curr_step + 1
        )
        return state
    
    def active_players(state):
        return active_family_tree(state.family_tree)
    
    def family_info(next_state):
        birthdays = next_state.family_tree.player_list.players[...,0]
        current_time = next_state.family_tree.player_list.current_time 
        child_locations, = jnp.nonzero(
            birthdays == current_time,
            size=params.max_players,
            fill_value=params.max_players,
        )
        parent_info = next_state.family_tree.parents[child_locations]
        parent_locations = parent_info[...,1]
        
        return parent_locations, child_locations

    return population_game(
        init_state, transition, observe, active_players, family_info)
