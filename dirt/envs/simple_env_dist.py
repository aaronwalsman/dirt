#!/usr/bin/env python3
"""
Simplified Distributed Simple Environment with basic argument parsing.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
import argparse
from typing import Tuple, Any
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng
from jax import lax, pmap
import jax.tree_util as jtu

from mechagogue.static import static_data
from mechagogue.breed.normal import normal_mutate
from mechagogue.player_list import (
    make_birthday_player_list, make_player_family_tree)

from dirt.constants import (
    ROCK_COLOR, DEFAULT_BUG_COLOR, BIOMASS_AND_ENERGY_TINT, DEFAULT_FLOAT_DTYPE)
import dirt.gridworld2d.spawn as spawn
import dirt.gridworld2d.dynamics as dynamics
from dirt.gridworld2d.grid import (
    take_from_grid_locations, add_to_grid_locations)
from dirt.gridworld2d.observations import first_person_view

AXIS = "mesh"  # pmap axis name

# Actions from simple_env.py
NO_ACTION = 0
EAT_ACTION = 1
FORWARD_ACTION = 2
LEFT_ACTION = 3
RIGHT_ACTION = 4
REPRODUCE_ACTION = 5
NUM_ACTIONS = 6

# ---------------- Simple Parameters Class ----------------
@static_data
class SimpleEnvDistParams:
    # Core environment params
    initial_players : int = 1024  # per device
    max_players : int = 8192      # per device  
    world_size : Tuple[int,int] = (256, 256)  # per device interior (H, W)
    
    # Simple env specific params
    initial_food_density : float = 0.1
    per_step_food_density : float = 0.0005
    food_burn_rate : float = 0.01
    starting_food : float = 0.5
    
    # Distribution params
    ndev : int = 4
    halo : int = 3  # boundary exchange width
    
    # Training params
    seed : int = 1234
    steps : int = 5000
    learning_rate : float = 1e-2
    
    # Debug/test params
    test_mode : bool = False  # Simple test without full simulation
    allow_migration : bool = False  # Allow bugs to move between devices
    migration_freq : int = 10  # Migration frequency (every N steps)
    
    # Visualization
    save_video : bool = True
    video_path : str = './simple_env_distributed.mp4'
    video_fps : int = 30
    video_length : int = 30  # Target video length in seconds  
    max_video_frames : int = 1000

# ---------------- Halo Exchange for Food Grid ----------------
def make_food_halo_copy(H: int, W: int, halo: int, ndev: int):
    """
    Copy neighbor food border strips into local ghost cells.
    """
    L = halo
    R = halo + W
    
    perm_right_shift = [(i, (i + 1) % ndev) for i in range(ndev)]
    perm_left_shift  = [(i, (i - 1) % ndev) for i in range(ndev)]

    def halo_copy(food_grid: jnp.ndarray, role: jnp.ndarray) -> jnp.ndarray:
        dev = lax.axis_index(AXIS)

        # Real border strips (columns)  
        my_left_real  = food_grid[:, L : L + halo]
        my_right_real = food_grid[:, R - halo : R]

        # Exchange with neighbors
        payload_right = lax.cond(
            dev == (ndev - 1),
            lambda _: jnp.zeros_like(my_right_real),
            lambda _: my_right_real,
            operand=None,
        )
        left_neighbor_right = lax.ppermute(payload_right, axis_name=AXIS, perm=perm_right_shift)

        payload_left = lax.cond(
            dev == 0,
            lambda _: jnp.zeros_like(my_left_real),
            lambda _: my_left_real,
            operand=None,
        )
        right_neighbor_left = lax.ppermute(payload_left, axis_name=AXIS, perm=perm_left_shift)

        # Write ghosts (edge-guarded)
        food_grid = lax.cond(
            role > 0,
            lambda x: x.at[:, 0:L].set(left_neighbor_right),
            lambda x: x,
            operand=food_grid,
        )
        food_grid = lax.cond(
            role < (ndev - 1), 
            lambda x: x.at[:, R:R + halo].set(right_neighbor_left),
            lambda x: x,
            operand=food_grid,
        )
        return food_grid

    return halo_copy

# ---------------- Bug Migration System ----------------
def make_bug_migration_system(H: int, W: int, halo: int, ndev: int, max_players: int):
    """
    Enable actual bug migration between adjacent devices.
    """
    L = halo  # Left boundary of interior
    R = halo + W  # Right boundary of interior
    
    perm_right_shift = [(i, (i + 1) % ndev) for i in range(ndev)]
    perm_left_shift  = [(i, (i - 1) % ndev) for i in range(ndev)]

    def migrate_bugs(bug_x, bug_r, bug_stomach, active, models, role):
        """Process bug migration between devices"""
        dev = lax.axis_index(AXIS)
        
        # Identify bugs that want to migrate (in halo regions)
        migrate_left = (bug_x[:, 1] < L) & active  # In left halo
        migrate_right = (bug_x[:, 1] >= R) & active  # In right halo
        
        # Package migrant data
        def package_bug_data(mask, x_offset):
            # Only package data for bugs that are migrating
            migrant_x = jnp.where(mask, bug_x[:, 0], -1)  # y position (-1 for inactive)
            migrant_x_new = jnp.where(mask, bug_x[:, 1] + x_offset, -1)  # adjusted x position
            migrant_r = jnp.where(mask, bug_r, 0)
            migrant_stomach = jnp.where(mask, bug_stomach, 0.0)
            migrant_models = jnp.where(mask[:, None], models, 0.0)
            
            return jnp.stack([migrant_x, migrant_x_new, migrant_r, migrant_stomach], axis=1), migrant_models, mask
        
        # Package left and right migrants  
        left_data, left_models, left_mask = package_bug_data(migrate_left, W)  # Move to right side of left neighbor
        right_data, right_models, right_mask = package_bug_data(migrate_right, -W)  # Move to left side of right neighbor
        
        # Exchange with neighbors (corrected topology)
        # Send left migrants to left neighbor (device i-1)
        left_payload_data = lax.cond(dev == 0, lambda _: jnp.zeros_like(left_data), lambda _: left_data, operand=None)
        left_payload_models = lax.cond(dev == 0, lambda _: jnp.zeros_like(left_models), lambda _: left_models, operand=None)
        
        # Left migrants go to left neighbor, so they arrive as immigrants from the right
        incoming_from_right_data = lax.ppermute(left_payload_data, axis_name=AXIS, perm=perm_left_shift)
        incoming_from_right_models = lax.ppermute(left_payload_models, axis_name=AXIS, perm=perm_left_shift)
        
        # Send right migrants to right neighbor (device i+1)
        right_payload_data = lax.cond(dev == (ndev-1), lambda _: jnp.zeros_like(right_data), lambda _: right_data, operand=None)
        right_payload_models = lax.cond(dev == (ndev-1), lambda _: jnp.zeros_like(right_models), lambda _: right_models, operand=None)
        
        # Right migrants go to right neighbor, so they arrive as immigrants from the left  
        incoming_from_left_data = lax.ppermute(right_payload_data, axis_name=AXIS, perm=perm_right_shift)
        incoming_from_left_models = lax.ppermute(right_payload_models, axis_name=AXIS, perm=perm_right_shift)
        
        # Remove emigrants from current device
        stay_mask = ~(migrate_left | migrate_right)
        new_bug_x = jnp.where(stay_mask[:, None], bug_x, jnp.array([H+10, W+halo+10]))  # Move emigrants off-grid
        new_bug_r = jnp.where(stay_mask, bug_r, 0)
        new_bug_stomach = jnp.where(stay_mask, bug_stomach, 0.0)
        new_models = jnp.where(stay_mask[:, None], models, 0.0)
        new_active = stay_mask & active
        
        # Process incoming migrants
        def place_immigrants(incoming_data, incoming_models):
            # Find valid incoming bugs (y >= 0 indicates valid data)
            valid_incoming = incoming_data[:, 0] >= 0
            
            # Find available slots (inactive bugs)
            available_slots = ~new_active
            
            # Place immigrants in available slots
            def place_one_immigrant(i, carry):
                bug_x_c, bug_r_c, bug_stomach_c, models_c, active_c, available_slots_c = carry
                
                # Check if this immigrant should be placed
                immigrant_valid = valid_incoming[i]
                slot_available = jnp.sum(available_slots_c) > 0
                should_place = immigrant_valid & slot_available
                
                # Find first available slot
                slot_idx = jnp.argmax(available_slots_c)
                
                # Place immigrant
                immigrant_pos = jnp.array([incoming_data[i, 0], incoming_data[i, 1]], dtype=jnp.int32)
                bug_x_c = lax.cond(should_place, 
                                 lambda x: x.at[slot_idx].set(immigrant_pos),
                                 lambda x: x, bug_x_c)
                bug_r_c = lax.cond(should_place,
                                 lambda r: r.at[slot_idx].set(incoming_data[i, 2]),
                                 lambda r: r, bug_r_c)
                bug_stomach_c = lax.cond(should_place,
                                       lambda s: s.at[slot_idx].set(incoming_data[i, 3]),
                                       lambda s: s, bug_stomach_c)
                models_c = lax.cond(should_place,
                                  lambda m: m.at[slot_idx].set(incoming_models[i]),
                                  lambda m: m, models_c)
                active_c = lax.cond(should_place,
                                  lambda a: a.at[slot_idx].set(True),
                                  lambda a: a, active_c)
                
                # Update available slots
                available_slots_c = lax.cond(should_place,
                                           lambda a: a.at[slot_idx].set(False),
                                           lambda a: a, available_slots_c)
                
                return bug_x_c, bug_r_c, bug_stomach_c, models_c, active_c, available_slots_c
            
            # Process immigrants using JAX-compatible loop
            def process_immigrant(i, carry):
                return place_one_immigrant(i, carry)
            
            carry = (new_bug_x, new_bug_r, new_bug_stomach, new_models, new_active, available_slots)
            carry = lax.fori_loop(0, min(32, max_players), process_immigrant, carry)
            
            new_bug_x, new_bug_r, new_bug_stomach, new_models, new_active, _ = carry
            return new_bug_x, new_bug_r, new_bug_stomach, new_models, new_active
        
        # Step 1: Remove emigrants (bugs that moved to halo regions)
        stay_mask = ~(migrate_left | migrate_right)
        
        # Keep only staying bugs by masking their active status
        new_active = active & stay_mask
        
        # Clear data for emigrated bugs
        new_bug_x = jnp.where(stay_mask[:, None], bug_x, jnp.array([0, 0]))
        new_bug_r = jnp.where(stay_mask, bug_r, 0)
        new_bug_stomach = jnp.where(stay_mask, bug_stomach, 0.0)
        new_models = jnp.where(stay_mask[:, None], models, 0.0)
        
        # Step 2: Place immigrants in available slots (simplified)
        available_slots = ~new_active
        
        # Process left immigrants (place first valid one)
        valid_left = jnp.all(incoming_from_left_data != -1, axis=1)
        has_left_immigrant = jnp.any(valid_left)
        has_available_slot = jnp.any(available_slots)
        can_place_left = has_left_immigrant & has_available_slot
        
        # Find first valid immigrant and first available slot
        first_immigrant_idx = jnp.argmax(valid_left)
        first_slot_idx = jnp.argmax(available_slots)
        
        # Place the immigrant
        new_bug_x = lax.cond(
            can_place_left,
            lambda x: x.at[first_slot_idx].set(jnp.array([incoming_from_left_data[first_immigrant_idx, 0], incoming_from_left_data[first_immigrant_idx, 1]], dtype=jnp.int32)),
            lambda x: x,
            new_bug_x
        )
        new_bug_r = lax.cond(
            can_place_left,
            lambda r: r.at[first_slot_idx].set(incoming_from_left_data[first_immigrant_idx, 2]),
            lambda r: r,
            new_bug_r
        )
        new_bug_stomach = lax.cond(
            can_place_left,
            lambda s: s.at[first_slot_idx].set(incoming_from_left_data[first_immigrant_idx, 3]),
            lambda s: s,
            new_bug_stomach
        )
        new_models = lax.cond(
            can_place_left,
            lambda m: m.at[first_slot_idx].set(incoming_from_left_models[first_immigrant_idx]),
            lambda m: m,
            new_models
        )
        new_active = lax.cond(
            can_place_left,
            lambda a: a.at[first_slot_idx].set(True),
            lambda a: a,
            new_active
        )
        
        return new_bug_x, new_bug_r, new_bug_stomach, new_models, new_active
    
    return migrate_bugs

# ---------------- Simplified Environment ----------------
def make_simple_env_distributed(params: SimpleEnvDistParams):
    
    H, W = params.world_size
    halo = params.halo
    ndev = params.ndev
    
    def make_device_env(device_id: int):
        BUG_LOCATION_OFFSET = device_id * params.max_players
        player_list = make_birthday_player_list(
            params.max_players, location_offset=BUG_LOCATION_OFFSET)
        player_family_tree = make_player_family_tree(player_list)
        
        food_halo_copy = make_food_halo_copy(H, W, halo, ndev)
        bug_migration = make_bug_migration_system(H, W, halo, ndev, params.max_players)
        
        @static_data
        class SimpleEnvDistState:
            bug_x : jax.Array
            bug_r : jax.Array
            bug_object_grid : jax.Array
            bug_stomach : jax.Array
            family_tree_state : Any  # player_family_tree.State
            food : jax.Array
            reproduced : jax.Array
        
        def init_local_state(key: jax.Array, role: jax.Array):
            """Initialize state for one device"""
            
            family_tree_state = player_family_tree.init(params.max_players)
            active = player_family_tree.active(family_tree_state)
            
            # Only activate the first initial_players
            active = active.at[params.initial_players:].set(False)
            
            # Initialize bugs in interior region only [halo, halo+W)
            key, xr_key = jrng.split(key)
            interior_size = (H, W)
            active_slice = active[:params.initial_players]
            bug_x_active, bug_r_active = spawn.unique_xr(
                xr_key, params.initial_players, interior_size, active_slice)
            
            # Pad to full size
            bug_x = jnp.zeros((params.max_players, 2), dtype=jnp.int32)
            bug_r = jnp.zeros(params.max_players, dtype=jnp.int32)
            bug_x = bug_x.at[:params.initial_players].set(bug_x_active)
            bug_r = bug_r.at[:params.initial_players].set(bug_r_active)
            
            # Offset bug positions to interior region
            bug_x = bug_x + jnp.array([0, halo])
            
            # Create object grid with halo columns
            world_with_halo = (H, W + 2 * halo)
            bug_object_grid = dynamics.make_object_grid(
                world_with_halo, bug_x, active)
            
            bug_stomach = jnp.zeros(params.max_players)
            bug_stomach = bug_stomach.at[:params.initial_players].set(
                params.starting_food)
            
            # Initialize food in full grid (including halos) 
            key, food_key = jrng.split(key)
            n = round(
                params.initial_food_density *
                world_with_halo[0] * world_with_halo[1]
            )
            food = spawn.poisson_grid(
                food_key, n, n*2, world_with_halo)
            food = food.astype(jnp.float32)
            
            reproduced = jnp.zeros(params.max_players, dtype=jnp.bool)
            
            return SimpleEnvDistState(
                bug_x=bug_x,
                bug_r=bug_r, 
                bug_object_grid=bug_object_grid,
                bug_stomach=bug_stomach,
                family_tree_state=family_tree_state,
                food=food,
                reproduced=reproduced,
            )
        
        def transition_with_halo(
            key: jax.Array,
            state: SimpleEnvDistState, 
            action: jax.Array,
            traits: None,
            role: jax.Array,
        ):
            """Full environment step with halo exchange and reproduction"""
            
            # 1) Exchange food halos first
            food = food_halo_copy(state.food, role)
            
            # 2) Run full local environment dynamics (same as original simple_env.py)
            active = player_family_tree.active(state.family_tree_state)
            bug_x = state.bug_x
            bug_r = state.bug_r
            bug_object_grid = state.bug_object_grid
            bug_stomach = state.bug_stomach
            family_tree_state = state.family_tree_state
            
            # Eat and metabolism
            eat = (action == EAT_ACTION) & active
            food, edible_food = take_from_grid_locations(food, bug_x, 1., 1)
            consumed_food = eat * edible_food
            leftover_food = (~eat) * edible_food
            bug_stomach = bug_stomach + consumed_food
            food = add_to_grid_locations(food, bug_x, leftover_food, 1)
            bug_stomach -= params.food_burn_rate * active
            
            # Add new food
            key, food_key = jrng.split(key)
            world_with_halo = (H, W + 2 * halo)
            n = round(
                params.per_step_food_density *
                world_with_halo[0] * world_with_halo[1]
            )
            new_food = spawn.poisson_grid(      
                food_key, n, n*2, world_with_halo)
            food = food + new_food
            food = jnp.clip(food, min=0., max=1.)
            
            # Move bugs
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
            
            # Handle boundary crossings 
            L, R = halo, halo + W
            
            def handle_movement_boundaries(new_x, old_x):
                # Vertical constraints remain the same
                y_constrained = jnp.clip(new_x[:, 0], 0, H-1)
                x_pos = new_x[:, 1]
                
                if params.allow_migration:
                    # Allow movement into halo regions (step 1 towards full migration)
                    # This allows bugs to move closer to boundaries and interact across them
                    x_constrained = jnp.clip(x_pos, 0, W + 2*halo - 1)
                else:
                    # Allow bugs to reach the rightmost interior position (R-1)
                    # Bugs trying to go further get stuck at the boundary
                    x_constrained = jnp.clip(x_pos, L, R - 1)
                
                return jnp.stack([y_constrained, x_constrained], axis=1)
            
            bug_x = handle_movement_boundaries(bug_x, state.bug_x)
            
            # Reproduction
            wants_to_reproduce = (action == REPRODUCE_ACTION) & active
            able_to_reproduce = (bug_stomach > 1.) & active
            will_reproduce = wants_to_reproduce & able_to_reproduce
            
            # Determine new child locations, and filter will_reproduce based on
            # availability of those locations
            will_reproduce, child_x, child_r = spawn.spawn_from_parents(
                will_reproduce,
                bug_x,
                bug_r,
                object_grid=bug_object_grid,
            )
            
            # Handle child boundary crossings  
            child_x = handle_movement_boundaries(child_x, bug_x)
            
            # Filter will reproduce based on available slots
            available_slots = params.max_players - active.sum()
            will_reproduce &= jnp.cumsum(will_reproduce) <= available_slots
            
            # Charge parents
            bug_stomach -= will_reproduce * params.starting_food
            
            # Update family tree
            parent_locations, = jnp.nonzero(
                will_reproduce,
                size=params.max_players,
                fill_value=params.max_players,
            )
            
            # Compute deaths
            still_alive = bug_stomach > 0.
            recent_deaths = active & ~still_alive
            family_tree_state, child_locations, _ = player_family_tree.step(
                family_tree_state,
                recent_deaths,
                parent_locations[..., None],
            )
            active = player_family_tree.active(family_tree_state)
            
            # Update child resources
            bug_stomach = bug_stomach.at[child_locations].set(params.starting_food)
            
            # Move dead bugs off the map
            bug_x = jnp.where(
                recent_deaths[:,None],
                jnp.array([H, W + 2*halo], dtype=jnp.int32),  # Move off-grid
                bug_x,
            )
            bug_r = bug_r * active
            
            # Update the bug positions, rotations and object_grid with child data
            child_x = child_x[parent_locations]
            child_r = child_r[parent_locations]
            bug_x = bug_x.at[child_locations].set(child_x)
            bug_r = bug_r.at[child_locations].set(child_r)
            
            # Rebuild object grid
            world_with_halo = (H, W + 2 * halo)
            bug_object_grid = jnp.full(world_with_halo, -1, dtype=jnp.int32)
            bug_object_grid = bug_object_grid.at[bug_x[...,0], bug_x[...,1]].set(
                jnp.where(active, jnp.arange(params.max_players), -1))
            
            # Update state
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
        
        def observe(key: jax.Array, state: SimpleEnvDistState):
            """Create first-person observations for each bug"""
            image = render(state)
            return first_person_view(state.bug_x, state.bug_r, image, 3, 1, 1)
        
        def active_players(state: SimpleEnvDistState):
            return player_family_tree.active(state.family_tree_state)
        
        def family_info(next_state: SimpleEnvDistState):
            """Get parent-child relationships for the current timestep"""
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
            """Render only the interior region for each device"""
            L, R = halo, halo + W
            interior_food = state.food[:, L:R]
            interior_size = (H, W)
            
            image = jnp.full((*interior_size, 3), ROCK_COLOR, dtype=jnp.float32)
            
            # Filter bugs to interior region
            active = active_players(state)
            interior_bugs = (state.bug_x[:, 1] >= L) & (state.bug_x[:, 1] < R) & active
            interior_bug_x = state.bug_x - jnp.array([0, halo])
            
            bug_color = jnp.where(
                state.reproduced[...,None], jnp.ones(3), DEFAULT_BUG_COLOR)
            
            image = image + interior_food[...,None] * BIOMASS_AND_ENERGY_TINT
            
            # Simple rendering - just add bug colors where bugs exist
            # This avoids complex conditional logic
            valid_positions = (
                (interior_bug_x[:, 0] >= 0) & (interior_bug_x[:, 0] < H) &
                (interior_bug_x[:, 1] >= 0) & (interior_bug_x[:, 1] < W) &
                interior_bugs
            )
            
            # Use scatter-add for rendering bugs
            bug_indices = jnp.where(valid_positions[:, None], interior_bug_x, jnp.array([H-1, W-1]))
            image = image.at[bug_indices[:, 0], bug_indices[:, 1]].add(
                jnp.where(valid_positions[:, None], bug_color * 0.5, 0.0))
            
            return jnp.clip(image, 0.0, 1.0)
        
        return init_local_state, transition_with_halo, observe, active_players, family_info, render, bug_migration
    
    return make_device_env

def stitch_global_image(sharded_renders, ndev: int):
    """Combine rendered images from all devices into global view"""
    host_renders = [jax.device_get(sharded_renders[i]) for i in range(ndev)]
    return np.concatenate(host_renders, axis=1)

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Simple Environment')
    parser.add_argument('--world_height', type=int, default=256, help='Height per device')
    parser.add_argument('--world_width', type=int, default=256, help='Width per device')
    parser.add_argument('--ndev', type=int, default=4, help='Number of devices')
    parser.add_argument('--halo', type=int, default=3, help='Halo width')
    parser.add_argument('--steps', type=int, default=5000, help='Simulation steps')
    parser.add_argument('--initial_players', type=int, default=1024, help='Initial players per device')
    parser.add_argument('--max_players', type=int, default=8192, help='Max players per device')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Mutation learning rate')
    parser.add_argument('--initial_food_density', type=float, default=0.1, help='Initial food density')
    parser.add_argument('--per_step_food_density', type=float, default=0.0005, help='Food added per step')
    parser.add_argument('--food_burn_rate', type=float, default=0.01, help='Food burn rate per step')
    parser.add_argument('--starting_food', type=float, default=0.5, help='Starting food per bug')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--allow_migration', action='store_true', help='Allow bugs to move between devices')
    parser.add_argument('--migration_freq', type=int, default=10, help='Migration frequency (every N steps)')
    parser.add_argument('--test_mode', action='store_true', help='Test mode - init only')
    parser.add_argument('--save_video', action='store_true', default=True, help='Save video')
    parser.add_argument('--no_video', action='store_true', help='Disable video generation')
    parser.add_argument('--video_path', type=str, default='./simple_env_distributed.mp4', help='Video path')
    parser.add_argument('--video_fps', type=int, default=30, help='Video framerate')
    parser.add_argument('--video_length', type=int, default=30, help='Target video length in seconds')
    parser.add_argument('--max_video_frames', type=int, default=1000, help='Maximum video frames to save')
    return parser.parse_args()

def main():
    print("=== Simplified Distributed Simple Environment ===")
    
    args = parse_args()
    
    # Validate parameters
    world_positions = args.world_height * args.world_width
    if args.initial_players >= world_positions:
        print(f"ERROR: initial_players ({args.initial_players}) >= world positions ({world_positions})")
        print(f"Adjusting initial_players to {world_positions // 4}")
        args.initial_players = world_positions // 4
    
    # Handle video settings
    if args.no_video:
        args.save_video = False
    
    # Create params from args
    params = SimpleEnvDistParams(
        world_size=(args.world_height, args.world_width),
        ndev=args.ndev,
        halo=args.halo,
        steps=args.steps,
        initial_players=args.initial_players,
        max_players=args.max_players,
        learning_rate=args.learning_rate,
        initial_food_density=args.initial_food_density,
        per_step_food_density=args.per_step_food_density,
        food_burn_rate=args.food_burn_rate,
        starting_food=args.starting_food,
        seed=args.seed,
        allow_migration=args.allow_migration,
        migration_freq=args.migration_freq,
        test_mode=args.test_mode,
        save_video=args.save_video,
        video_path=args.video_path,
        video_fps=args.video_fps,
        video_length=args.video_length,
        max_video_frames=args.max_video_frames,
    )
    
    H, W = params.world_size
    ndev = params.ndev
    
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.device_count()}")
    print(f"Device list: {jax.devices()}")
    
    assert jax.device_count() >= ndev, f"Need >= {ndev} devices; found {jax.device_count()}"
    devices = jax.devices()[:ndev]
    print(f"Using {ndev} devices:", devices)
    print(f"Each device: {H}x{W} interior + {params.halo} halo")
    print(f"Global domain: {H}x{W*ndev}")
    print("=== Starting initialization ===")
    
    # Create distributed environment
    print("Creating distributed environment functions...")
    make_device_env = make_simple_env_distributed(params)
    roles = jnp.arange(ndev, dtype=jnp.int32)
    
    # Get device-specific functions
    print("Getting device-specific functions...")
    device_fns = [make_device_env(i) for i in range(ndev)]
    init_fn, transition_fn, observe_fn, active_fn, family_fn, render_fn, migrate_fn = device_fns[0]
    print("Device functions created successfully")
    
    # pmap the functions
    print("Creating pmap functions...")
    init_pm = pmap(init_fn, in_axes=(0, 0), axis_name=AXIS, devices=devices)
    transition_pm = pmap(transition_fn, in_axes=(0, 0, 0, None, 0), axis_name=AXIS, devices=devices)
    observe_pm = pmap(observe_fn, in_axes=(0, 0), axis_name=AXIS, devices=devices)
    render_pm = pmap(render_fn, in_axes=0, axis_name=AXIS, devices=devices)
    print("All pmap functions created")
    
    # Initialize
    key = jrng.key(params.seed)
    keys = jrng.split(key, ndev + 1)
    key, init_keys = keys[0], keys[1:]
    init_keys = jnp.stack(init_keys, axis=0)
    
    print("Initializing distributed environment...")
    print("This may take a while for JIT compilation...")
    sharded_state = init_pm(init_keys, roles)
    print("SUCCESS: Distributed environment initialized!")
    
    # Test mode: just verify initialization and exit
    if params.test_mode:
        print("Test mode: initialization successful, exiting...")
        test_state = jax.device_get(jtu.tree_map(lambda x: x[0], sharded_state))
        print("State shapes:")
        print(f"  bug_x: {test_state.bug_x.shape}")
        print(f"  food: {test_state.food.shape}")
        print(f"  bug_stomach: {test_state.bug_stomach.shape}")
        
        # Test rendering
        print("Testing render...")
        sharded_renders = render_pm(sharded_state)
        global_image = stitch_global_image(sharded_renders, ndev)
        print(f"Global image shape: {global_image.shape}")
        print("SUCCESS: Test mode complete!")
        return
    
    print(f"Running {params.steps} distributed steps...")
    
    import time
    start_time = time.time()
    
    # Initialize models (neural network weights for each bug)
    key, model_key = jrng.split(key)
    model_keys = jrng.split(model_key, ndev)
    models = []
    for i in range(ndev):
        model = jrng.normal(model_keys[i], (params.max_players, NUM_ACTIONS))
        models.append(model) 
    sharded_models = jnp.stack(models, axis=0)
    
    # Evolution parameters
    mutate = normal_mutate(learning_rate=params.learning_rate)
    
    # Action function using neural network models
    def act(key, model, obs):
        """Gumbel-softmax action selection"""
        u = jrng.uniform(key, model.shape, minval=1e-6, maxval=1.)
        gumbel = -jnp.log(-jnp.log(u))
        actions = jnp.argmax(model + gumbel, axis=-1)
        return actions
    
    act_pm = pmap(act, in_axes=(0, 0, 0), axis_name=AXIS, devices=devices)
    
    # Video recording setup
    if params.save_video:
        import imageio
        video_frames = []
        
        # Calculate sampling frequency for desired video length
        target_frames = min(params.video_fps * params.video_length, params.max_video_frames)
        video_sample_freq = max(1, params.steps // target_frames)
        print(f"Video sampling: every {video_sample_freq} steps for {target_frames} frames ({target_frames/params.video_fps:.1f}s at {params.video_fps}fps)")
    
    # Get initial observations
    key, obs_key = jrng.split(key)
    obs_keys = jrng.split(obs_key, ndev)
    sharded_obs = observe_pm(obs_keys, sharded_state)
    
    # Define step function for evolutionary updates
    def update_models(mutate_key, models, state, next_state):
        """Update models based on reproduction events"""
        parents, children = family_fn(next_state)
        
        # Mutate parent models for children
        parent_models = models[parents]
        mutated_parents = mutate(mutate_key, parent_models)
        
        # Update child models with mutated parent models  
        next_models = models.at[children].set(mutated_parents[..., 0])
        
        return next_models
    
    update_models_pm = pmap(update_models, in_axes=(0, 0, 0, 0), axis_name=AXIS, devices=devices)
    
    # Run simulation with evolution
    for step in range(params.steps):
        # Get actions from neural network models
        key, action_key = jrng.split(key)
        action_keys = jrng.split(action_key, ndev)
        sharded_actions = act_pm(action_keys, sharded_models, sharded_obs)
        
        # Environment step
        key, env_key = jrng.split(key)
        env_keys = jrng.split(env_key, ndev)
        sharded_next_state = transition_pm(
            env_keys, sharded_state, sharded_actions, None, roles)
        
        # Bug migration step (if enabled and at right frequency)
        if params.allow_migration and (step % params.migration_freq == 0):
            def migrate_step(state, models, role):
                active = active_fn(state)
                new_bug_x, new_bug_r, new_bug_stomach, new_models, new_active = migrate_fn(
                    state.bug_x, state.bug_r, state.bug_stomach, active, models, role)
                
                # Update family tree to reflect new active status
                # For now, keep it simple - just update positions
                new_state = state.replace(
                    bug_x=new_bug_x,
                    bug_r=new_bug_r, 
                    bug_stomach=new_bug_stomach
                )
                return new_state, new_models
            
            migrate_pm = pmap(migrate_step, in_axes=(0, 0, 0), axis_name=AXIS, devices=devices)
            sharded_next_state, sharded_models = migrate_pm(sharded_next_state, sharded_models, roles)
        
        # Get next observations
        key, obs_key = jrng.split(key)
        obs_keys = jrng.split(obs_key, ndev)
        sharded_next_obs = observe_pm(obs_keys, sharded_next_state)
        
        # Update models based on reproduction events
        key, mutate_key = jrng.split(key)
        mutate_keys = jrng.split(mutate_key, ndev)
        sharded_models = update_models_pm(
            mutate_keys, sharded_models, sharded_state, sharded_next_state)
        
        # Update for next iteration
        sharded_state = sharded_next_state
        sharded_obs = sharded_next_obs
        
        # Render for video
        if params.save_video and step % video_sample_freq == 0:
            sharded_renders = render_pm(sharded_state)
            global_image = stitch_global_image(sharded_renders, ndev)
            video_frames.append(global_image)
        
        # Progress logging
        if (step + 1) % max(1, params.steps // 10) == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"Step {step + 1}/{params.steps} ({steps_per_sec:.1f} steps/sec)")
    
    print("Simulation complete!")
    
    # Save video
    if params.save_video and video_frames:
        print(f"Saving video to {params.video_path}...")
        video_frames = np.array(video_frames)
        video_frames = (video_frames * 255).astype(np.uint8)
        video_frames = video_frames[:,::-1]  # Flip vertically
        
        import imageio
        writer = imageio.get_writer(
            params.video_path,
            fps=params.video_fps,
            codec='libx264', 
            ffmpeg_params=['-crf', '18', '-preset', 'slow'],
        )
        
        with writer:
            for frame in video_frames:
                writer.append_data(frame)
        
        print(f"Video saved: {params.video_path}")
    
    # Final statistics
    host_states = [jax.device_get(jtu.tree_map(lambda x: x[i], sharded_state)) for i in range(ndev)]
    total_active = sum(np.sum(active_fn(s)) for s in host_states)
    
    print(f"Final stats: {total_active} total active players")

if __name__ == '__main__':
    main()



# python simple_env_dist.py \
#   --world_height 256 \
#   --world_width 256 \
#   --ndev 4 \
#   --halo 3 \
#   --steps 10000 \
#   --initial_players 1024 \
#   --max_players 8192 \
#   --learning_rate 0.01 \
#   --initial_food_density 0.1 \
#   --per_step_food_density 0.0005 \
#   --food_burn_rate 0.01 \
#   --starting_food 0.5 \
#   --seed 1234 \
#   --allow_migration \
#   --migration_freq 10 \
#   --save_video \
#   --video_path ./my_simulation.mp4 \
#   --video_fps 30 \
#   --video_length 60 \
#   --max_video_frames 1800


# python simple_env_dist.py   --world_height 256   --world_width 256   --ndev 4   --halo 3   --steps 10000   --initial_players 1024   --max_players 8192   --learning_rate 0.01   --initial_food_density 0.1   --per_step_food_density 0.0005   --food_burn_rate 0.01   --starting_food 0.5   --seed 1234   --allow_migration   --migration_freq 10   --save_video   --video_path ./my_simulation.mp4   --video_fps 30   --video_length 60   --max_video_frames 1800