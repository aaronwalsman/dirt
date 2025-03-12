import jax
import jax.numpy as jnp
import jax.random as jrng
import chex

from mechagogue.dp.population_game import population_game
from mechagogue.player_list import birthday_player_list, player_family_tree
from mechagogue.pop.natural_selection import NaturalSelectionState

from dirt.gridworld2d import dynamics, observations, spawn
from dirt.examples.nomnom.nomnom_env import nomnom, NomNomParams, NomNomState, NomNomAction, NomNomObservation
from dirt.examples.nomnom.nomnom_model import nomnom_model, NomNomModelParams

# Reuse the existing NomNomParams but create a new instance with customized values.
custom_5x5_params_fixed_food = NomNomParams(
    world_size=(5, 5),
    initial_players=1,          
    max_players=1,               
    mean_initial_food=0,         
    max_initial_food=0,
    mean_food_growth=0,
    max_food_growth=0,
    
    # Metabolism/energy parameters
    initial_energy=1.0,          
    max_energy=5.0,
    food_metabolism=1.0,
    move_metabolism=-0.05,
    wait_metabolism=-0.025,
    senescence=0.0,
    
    view_width=5,
    view_distance=5
)

def nomnom_no_reproduce(params: NomNomParams = custom_5x5_params_fixed_food):
    """
    Returns an environment that starts in a 5x5 grid, 
    places 1 agent at the top edge, and ignores reproduction actions.
    """
    init_players, step_players, active_players = birthday_player_list(params.max_players)
    init_family_tree, step_family_tree, active_family_tree = player_family_tree(
        init_players, step_players, active_players, 1
    )

    def init_state_no_reproduce(
        key: chex.PRNGKey,
    ) -> NomNomState:
        """
        Creates a NomNomState for a single-agent 5x5 environment with the agent 
        on the top edge facing 'down' (or whichever direction you choose).
        """
        # Initialize the players
        family_tree = init_family_tree(params.initial_players)
        active_players = active_family_tree(family_tree)
        
        # place it at the top row, column 2 => (0,2)
        player_x = jnp.array([[0, 2]])
        player_r = jnp.array([2])
        
        # keep them generalized
        maxp = params.max_players
        full_player_x = jnp.zeros((maxp, 2), dtype=jnp.int32)
        full_player_r = jnp.zeros((maxp,), dtype=jnp.int32)
        full_player_x = full_player_x.at[0].set(player_x[0])
        full_player_r = full_player_r.at[0].set(player_r[0])
        
        # For energy, age, etc.
        player_energy = jnp.zeros((maxp,))
        player_energy = player_energy.at[0].set(params.initial_energy)
        player_age = jnp.zeros((maxp,), dtype=jnp.int32)
        
        # Make an object grid (no collisions with just one player)
        object_grid = dynamics.make_object_grid(params.world_size, full_player_x, active_players)
        
        # Make a food grid
        key, foodkey = jrng.split(key)
        food_grid = spawn.poisson_grid(
            foodkey,
            params.mean_initial_food,
            params.max_initial_food,
            params.world_size,
        )
        
        # Build the state
        state = NomNomState(
            food_grid,
            object_grid,
            family_tree,
            player_x,
            player_r,
            player_energy,
            player_age,
            0
        )
        
        return state

    def transition_no_reproduce(
        key: chex.PRNGKey,
        state: NomNomState,
        action: NomNomAction,
    ) -> NomNomState:
        """
        A specialized transition that ignores the 'reproduce' action.
        """
        # Force reproduction to 0
        no_repro_action = NomNomAction(
            forward=action.forward,
            rotate=action.rotate,
            reproduce=jnp.zeros_like(action.reproduce)
        )
        
        # Then perform the usual steps
        active_players = active_family_tree(state.family_tree)
        
        # Move & rotate
        player_x, player_r, _, object_grid = dynamics.forward_rotate_step(
            state.player_x,
            state.player_r,
            no_repro_action.forward,
            no_repro_action.rotate,
            active=active_players,
            check_collisions=True,
            object_grid=state.object_grid
        )
        
        # age
        player_age = (state.player_age + active_players) * active_players
        
        # eat
        food_at_player = state.food_grid[player_x[...,0], player_x[...,1]]
        eaten_food = food_at_player * active_players
        player_energy = jnp.clip(
            state.player_energy + eaten_food * params.food_metabolism,
            0,
            params.max_energy,
        )
        food_grid = state.food_grid.at[player_x[...,0], player_x[...,1]].set(
            food_at_player & jnp.logical_not(eaten_food.astype(jnp.int32))
        )
        
        # metabolism
        moved = no_repro_action.forward | (no_repro_action.rotate != 0)
        energy_consumption = (
            moved * params.move_metabolism +
            (1. - moved) * params.wait_metabolism
        )
        energy_consumption *= (1. + params.senescence)**player_age
        player_energy = (player_energy + energy_consumption) * active_players
        
        # starvation
        deaths = player_energy <= 0.0

        # Same structure
        reproduce = (
            action.reproduce &
            (player_energy > params.initial_energy) &
            active_players
        )
        
        # Family updates
        n = reproduce.shape[0]
        parent_locations, = jnp.nonzero(reproduce, size=n, fill_value=n)
        parent_locations = parent_locations[...,None]
        family_tree, child_locations = step_family_tree(
            state.family_tree, deaths, parent_locations)
        
        # object_grid updates for dead players
        object_grid = object_grid.at[player_x[...,0], player_x[...,1]].set(
            jnp.where(deaths, -1, jnp.arange(params.max_players))
        )
        player_x = jnp.where(
            deaths[:,None],
            jnp.array(params.world_size, dtype=jnp.int32),
            player_x
        )
        player_r = jnp.where(deaths, 0, player_r)
        
        # food growth
        key, food_key = jrng.split(key)
        new_food = spawn.poisson_grid(
            food_key,
            params.mean_food_growth,
            params.max_food_growth,
            params.world_size,
        )
        food_grid = food_grid | new_food
        
        # next state
        next_state = NomNomState(
            food_grid,
            object_grid,
            family_tree,
            player_x,
            player_r,
            player_energy,
            player_age,
            state.curr_step + 1
        )
        return next_state

    def observe_no_reproduce(
        key: chex.PRNGKey,
        state: NomNomState,
    ) -> NomNomObservation:
        """
        Simple observation that includes a local 'view' of the grid and 
        the agent's current energy.
        """
        view_grid = state.food_grid.astype(jnp.uint8)
        active_players = active_family_tree(state.family_tree)
        view_grid = view_grid.at[state.player_x[...,0], state.player_x[...,1]].set(
            2 * active_players
        )
        
        view = observations.first_person_view(
            state.player_x,
            state.player_r,
            view_grid,
            params.view_width,
            params.view_distance,
            out_of_bounds=3,
        )
        
        return NomNomObservation(view, state.player_energy)
    
    def active_players_no_reproduce(state):
        return active_family_tree(state.family_tree)
    
    def family_info_no_reproduce(next_state):
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

    # Return a population_game environment
    return population_game(
        init_state_no_reproduce,
        transition_no_reproduce,
        observe_no_reproduce,
        active_players_no_reproduce,
        family_info_no_reproduce
    )


def place_food_in_middle(state: NomNomState) -> NomNomState:
    """
    Sets a single piece of food in the center (2,2) of a 5x5 grid.
    """
    updated_food_grid = state.food_grid.at[2, 2].set(1)
    
    # Build a new state with the updated food grid
    new_state = NomNomState(
        food_grid=updated_food_grid,
        object_grid=state.object_grid,
        family_tree=state.family_tree,
        player_x=state.player_x,
        player_r=state.player_r,
        player_energy=state.player_energy,
        player_age=state.player_age,
        curr_step=state.curr_step
    )
    return new_state