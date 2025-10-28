from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn

import imageio

from mechagogue.static import static_data, static_functions
from mechagogue.breed.normal import normal_mutate
from mechagogue.player_list import (
    make_birthday_player_list, make_player_family_tree)
from mechagogue.dp.poeg import make_poeg

from dirt.constants import (
    ROCK_COLOR, DEFAULT_BUG_COLOR, BIOMASS_AND_ENERGY_TINT)
import dirt.gridworld2d.spawn as spawn
import dirt.gridworld2d.dynamics as dynamics
from dirt.gridworld2d.grid import (
    take_from_grid_locations, add_to_grid_locations)
from dirt.gridworld2d.observations import first_person_view

@static_data
class SimpleEnvParams:
    initial_players : int = 1024
    max_players : int = 8192
    world_size : Tuple[int, int] = (256,256)
    
    initial_food_density : float = 0.1
    per_step_food_density : float = 0.0005
    food_burn_rate : float = 0.01
    starting_food : float = 0.5

NO_ACTION = 0
EAT_ACTION = 1
FORWARD_ACTION = 2
LEFT_ACTION = 3
RIGHT_ACTION = 4
REPRODUCE_ACTION = 5
NUM_ACTIONS = 6

def make_simple_env(params : SimpleEnvParams):
    
    # BUG_LOCATION_OFFSET should be set to GPU_ID * params.max_players
    BUG_LOCATION_OFFSET = 0
    player_list = make_birthday_player_list(
        params.max_players, location_offset=BUG_LOCATION_OFFSET)
    player_family_tree = make_player_family_tree(player_list)
    
    @static_data
    class SimpleEnvState:
        bug_x : jax.Array
        bug_r : jax.Array
        bug_object_grid : jax.Array
        bug_stomach : jax.Array
        
        family_tree_state : player_family_tree.State
        
        food : jax.Array
        reproduced : jax.Array
    
    def init_state(
        key : jax.Array
    ):
        
        family_tree_state = player_family_tree.init(params.initial_players)
        active = player_family_tree.active(family_tree_state)
        
        key, xr_key = jrng.split(key)
        bug_x, bug_r = spawn.unique_xr(
            xr_key, params.max_players, params.world_size, active)
        bug_object_grid = dynamics.make_object_grid(
            params.world_size, bug_x, active)
        
        bug_stomach = jnp.zeros(params.max_players)
        bug_stomach = bug_stomach.at[:params.initial_players].set(
            params.starting_food)
        
        key, food_key = jrng.split(key)
        n = round(
            params.initial_food_density *
            params.world_size[0] * params.world_size[1]
        )
        food = spawn.poisson_grid(
            food_key, n, n*2, params.world_size)
        food = food.astype(jnp.float32)
        
        reproduced = jnp.zeros(params.max_players, dtype=jnp.bool)
        
        return SimpleEnvState(
            bug_x,
            bug_r,
            bug_object_grid,
            bug_stomach,
            family_tree_state,
            food,
            reproduced,
        )
    
    def transition(
        key : jax.Array,
        state : SimpleEnvState,
        action : jax.Array,
        traits : None,
    ):
        # pull out variables
        active = active_players(state)
        bug_x = state.bug_x
        bug_r = state.bug_r
        bug_object_grid = state.bug_object_grid
        food = state.food
        bug_stomach = state.bug_stomach
        family_tree_state = state.family_tree_state
        
        # eat and metabolism
        # - which bugs will try to eat
        eat = (action == EAT_ACTION) & active
        # - pull edible food out of the environment
        food, edible_food = take_from_grid_locations(
            food, bug_x, 1., 1)
        # - determine which food will be consumed and which is leftover
        consumed_food = eat * edible_food
        leftover_food = (~eat) * edible_food
        # - feed the consumed food to the bugs
        bug_stomach = bug_stomach + consumed_food
        # - put the leftover food back in the environment
        food = add_to_grid_locations(food, bug_x, leftover_food, 1)
        # - burn food
        bug_stomach -= params.food_burn_rate * active
        
        # sprinkle more food into the environment
        key, food_key = jrng.split(key)
        n = round(
            params.per_step_food_density *
            params.world_size[0] * params.world_size[1]
        )
        new_food = spawn.poisson_grid(      
            food_key, n, n*2, params.world_size)
        food = food + new_food
        food = jnp.clip(food, min=0., max=1.)
        
        # move bugs
        forward = (action == FORWARD_ACTION).astype(jnp.int32)
        rotate = (
            (action == LEFT_ACTION).astype(jnp.int32) - 
            (action == RIGHT_ACTION).astype(jnp.int32)
        )
        bug_x, bug_r, _, bug_object_grid = dynamics.forward_rotate_step(
            bug_x,
            bug_r,
            forward,
            rotate,
            active=active,
            check_collisions=True,
            object_grid=bug_object_grid,
        )
        
        # reproduce
        # - determine which bugs will reproduce
        wants_to_reproduce = (action == REPRODUCE_ACTION) & active
        able_to_reproduce = (bug_stomach > 1.) & active
        will_reproduce = wants_to_reproduce & able_to_reproduce
        # - determine new child locations, and filter will_reproduce based on
        #   availability of those locations
        will_reproduce, child_x, child_r = spawn.spawn_from_parents(
            will_reproduce,
            bug_x,
            bug_r,
            object_grid=bug_object_grid,
        )
        # - filter will reproduce one more time based on how many available
        #   slots there are
        available_slots = params.max_players - active.sum()
        will_reproduce &= jnp.cumsum(will_reproduce) <= available_slots
        # - charge parents
        bug_stomach -= will_reproduce * params.starting_food
        # - update family tree
        parent_locations, = jnp.nonzero(
            will_reproduce,
            size=params.max_players,
            fill_value=params.max_players,
        )
        # - compute deaths
        still_alive = bug_stomach > 0.
        recent_deaths = active & ~still_alive
        family_tree_state, child_locations, _ = player_family_tree.step(
            family_tree_state,
            recent_deaths,
            parent_locations[..., None],
        )
        active = player_family_tree.active(family_tree_state)
        
        # update child resources
        bug_stomach = bug_stomach.at[child_locations].set(params.starting_food)
        
        # move
        # - move dead bugs off the map
        bug_x = jnp.where(
            recent_deaths[:,None],
            jnp.array(params.world_size, dtype=jnp.int32),
            bug_x,
        )
        bug_r = bug_r * active
        # - update the bug positions, rotations and object_grid with the
        #   child data
        child_x = child_x[parent_locations]
        child_r = child_r[parent_locations]
        bug_x = bug_x.at[child_locations].set(child_x)
        bug_r = bug_r.at[child_locations].set(child_r)
        bug_object_grid = jnp.full(params.world_size, -1, dtype=jnp.int32)
        bug_object_grid = bug_object_grid.at[bug_x[...,0], bug_x[...,1]].set(
            jnp.where(active, jnp.arange(params.max_players), -1))
        
        # update state
        state = state.replace(
            bug_x=bug_x,
            bug_r=bug_r,
            bug_object_grid=bug_object_grid,
            bug_stomach=bug_stomach,
            food=food,
            family_tree_state=family_tree_state,
            reproduced=will_reproduce,
        )
        
        return state
    
    def observe(
        key : jax.Array,
        state : SimpleEnvState,
    ):
        image = render(state)
        return first_person_view(state.bug_x, state.bug_r, image, 3, 1, 1)
    
    def active_players(
        state : SimpleEnvState,
    ):
        return player_family_tree.active(state.family_tree_state)
    
    def family_info(
        next_state : SimpleEnvState,
    ):
        birthdays = next_state.family_tree_state.player_state.players[...,0]
        current_time = next_state.family_tree_state.player_state.current_time
        child_locations, = jnp.nonzero(
            (birthdays == current_time) & (current_time != 0),
            size=params.max_players,
            fill_value=params.max_players,
        )
        parent_info = next_state.family_tree_state.parents[child_locations]
        parent_locations = parent_info[...,1] - BUG_LOCATION_OFFSET
        
        return parent_locations, child_locations
    
    def render(state):
        image = jnp.full((*params.world_size, 3), ROCK_COLOR, dtype=jnp.float32)
        bug_color = jnp.where(
            state.reproduced[...,None], jnp.ones(3), DEFAULT_BUG_COLOR)
        #image = jnp.where(
        #    state.food[...,None], 0.5 + BIOMASS_AND_ENERGY_TINT, image)
        image = image + state.food[...,None] * BIOMASS_AND_ENERGY_TINT
        image = image.at[state.bug_x[...,0], state.bug_x[...,1]].set(bug_color)
        
        return image
    
    game = make_poeg(
        init_state,
        transition,
        observe,
        active_players,
        family_info,
        render=render,
    )
    
    return game

def main():
    params = SimpleEnvParams(
        #initial_players=8,
        #max_players=16,
    )
    simple_env = make_simple_env(params)
    
    key = jrng.key(1234)
    steps = 5000
    
    key, init_key = jrng.split(key)
    state, obs, players = simple_env.init(init_key)
    
    key, model_key = jrng.split(key)
    model = jrng.normal(model_key, (params.max_players, NUM_ACTIONS))
    
    mutate = normal_mutate(learning_rate=1e-2)
    
    def act(key, model, obs):
        u = jrng.uniform(
            key,
            model.shape,
            minval=1e-6,
            maxval=1.,
        )
        gumbel = -jnp.log(-jnp.log(u))
        actions = jnp.argmax(model + gumbel, axis=-1)
        
        return actions
    
    def step(key_state_obs_model, _):
        # decompose inputs
        key, state, obs, model = key_state_obs_model
        
        # sample actions
        key, action_key = jrng.split(key)
        actions = act(key, model, obs)
        
        # take environment step
        key, env_key = jrng.split(key)
        next_state, next_obs, players, parents, children = simple_env.step(
            env_key, state, actions, None)
        
        # udpate model weights
        key, mutate_key = jrng.split(key)
        parent_models = model[parents]
        mutated_parents = mutate(mutate_key, parent_models)
        #mutated_parents = parent_models
        next_model = model.at[children].set(mutated_parents[...,0])
        
        # render the image for the video
        image = simple_env.render(next_state)
        
        return (key, next_state, next_obs, next_model), image
    
    (key, state, obs, model), video = jax.lax.scan(
        step,
        (key, state, obs, model),
        None,
        length=steps,
    )
    
    #===========================================================================
    # Don't distribute after here
    video = np.array((video * 255).astype(jnp.uint8))
    video = video[:,::-1]
    video_path = './simple_video.mp4'
    writer = imageio.get_writer(
        video_path,
        fps=30,
        codec='libx264',
        ffmpeg_params=[
            '-crf', '18',
            '-preset', 'slow',
        ],
    )
    with writer:
        n = video.shape[0]
        for i in range(n):
            writer.append_data(video[i])

if __name__ == '__main__':
    main()
