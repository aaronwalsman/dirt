#!/usr/bin/env python3
"""
Simplified Distributed Simple Environment.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Enable NCCL P2P over NVLink for GPU-to-GPU communication
os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
os.environ.setdefault("NCCL_SHM_DISABLE", "0")  # Allow shared memory for local GPUs

import sys
import argparse
from typing import Tuple, Any
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng
from jax import lax, pmap
import jax.tree_util as jtu
from jax.experimental.compilation_cache import compilation_cache as cc

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
    migration_freq : int = 1  # Migration frequency (every N steps)
    
    # Visualization
    save_video : bool = True
    video_path : str = './simple_env_distributed.mp4'
    video_fps : int = 30
    video_length : int = 30  # Target video length in seconds  
    max_video_frames : int = 1000
    
    # Performance/Diagnostics
    profile_communication : bool = False  # Profile GPU communication
    verify_gpu_direct : bool = True  # Verify direct GPU-to-GPU communication

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

# ---------------- Halo Exchange for Bug Object Grid ----------------
def make_bug_object_halo_copy(H: int, W: int, halo: int, ndev: int):
    """
    Copy neighbor bug_object_grid border strips into local ghost cells.
    This allows bugs to see neighbors across device boundaries for collision detection.
    """
    L = halo
    R = halo + W
    
    perm_right_shift = [(i, (i + 1) % ndev) for i in range(ndev)]
    perm_left_shift  = [(i, (i - 1) % ndev) for i in range(ndev)]

    def halo_copy(bug_grid: jnp.ndarray, role: jnp.ndarray) -> jnp.ndarray:
        dev = lax.axis_index(AXIS)

        # Real border strips (columns) - use -1 for empty cells
        my_left_real  = bug_grid[:, L : L + halo]
        my_right_real = bug_grid[:, R - halo : R]

        # Exchange with neighbors (use -1 for empty boundaries)
        payload_right = lax.cond(
            dev == (ndev - 1),
            lambda _: jnp.full_like(my_right_real, -1),
            lambda _: my_right_real,
            operand=None,
        )
        left_neighbor_right = lax.ppermute(payload_right, axis_name=AXIS, perm=perm_right_shift)

        payload_left = lax.cond(
            dev == 0,
            lambda _: jnp.full_like(my_left_real, -1),
            lambda _: my_left_real,
            operand=None,
        )
        right_neighbor_left = lax.ppermute(payload_left, axis_name=AXIS, perm=perm_left_shift)

        # Write ghosts (edge-guarded)
        bug_grid = lax.cond(
            role > 0,
            lambda x: x.at[:, 0:L].set(left_neighbor_right),
            lambda x: x,
            operand=bug_grid,
        )
        bug_grid = lax.cond(
            role < (ndev - 1), 
            lambda x: x.at[:, R:R + halo].set(right_neighbor_left),
            lambda x: x,
            operand=bug_grid,
        )
        return bug_grid

    return halo_copy

# ---------------- Bug Migration System ----------------
def make_bug_migration_system(H: int, W: int, halo: int, ndev: int, max_players: int, bug_object_halo_copy):
    """
    Enable actual bug migration between adjacent devices.
    Uses bug_object_grid halo exchange to check for collisions before migrating.
    Migration rules:
    - Bugs can only migrate to empty cells
    - If destination is occupied, bug stays on current device
    - Local moves have priority (handled by normal movement first)
    """
    L = halo  # Left boundary of interior
    R = halo + W  # Right boundary of interior
    
    perm_right_shift = [(i, (i + 1) % ndev) for i in range(ndev)]
    perm_left_shift  = [(i, (i - 1) % ndev) for i in range(ndev)]

    # Migration statistics counters
    migration_stats = jnp.zeros((ndev, 2), dtype=jnp.int32)  # [device, left/right]
    
    def migrate_bugs(bug_x, bug_r, bug_stomach, active, models, bug_object_grid, stats, role):
        """
        Process bug migration between devices with collision detection.
        
        Args:
            bug_x: Bug positions (max_players, 2)
            bug_r: Bug rotations (max_players,)
            bug_stomach: Bug food storage (max_players,)
            active: Bug active status (max_players,)
            models: Bug neural network weights (max_players, NUM_ACTIONS)
            bug_object_grid: Object grid with halo (H, W+2*halo)
            stats: Migration statistics array (ndev, 2)
            role: Device role (0 to ndev-1)
        
        Returns:
            Updated bug_x, bug_r, bug_stomach, models, active, bug_object_grid, stats
        """
        dev = lax.axis_index(AXIS)
        
        # First, exchange bug_object_grid halos to see neighbors
        bug_object_grid_with_neighbors = bug_object_halo_copy(bug_object_grid, role)
        
        # Identify bugs in halo regions that want to migrate
        migrate_left = (bug_x[:, 1] < L) & active  # In left halo
        migrate_right = (bug_x[:, 1] >= R) & active  # In right halo
        
        # For each migrating bug, check if destination cell is free
        # Left migrants: destination is in left neighbor's right interior border
        # Right migrants: destination is in right neighbor's left interior border
        
        # Vectorized collision checking
        # For bugs in left halo: check if their current cell is free in the global view
        # For bugs in right halo: check if their current cell is free in the global view
        left_dest_free = bug_object_grid_with_neighbors[bug_x[:, 0], bug_x[:, 1]] == -1
        right_dest_free = bug_object_grid_with_neighbors[bug_x[:, 0], bug_x[:, 1]] == -1
        
        can_migrate_left = migrate_left & left_dest_free
        can_migrate_right = migrate_right & right_dest_free
        
        # Package migrant data (only for bugs that CAN migrate)
        def package_migrants(can_migrate_mask, x_offset):
            """Package bug data for migration"""
            # Create migrant data arrays
            migrant_y = jnp.where(can_migrate_mask, bug_x[:, 0], -1)
            migrant_x = jnp.where(can_migrate_mask, bug_x[:, 1] + x_offset, -1)
            migrant_r = jnp.where(can_migrate_mask, bug_r, 0)
            migrant_stomach = jnp.where(can_migrate_mask, bug_stomach, 0.0)
            migrant_models = jnp.where(can_migrate_mask[:, None], models, 0.0)
            
            return migrant_y, migrant_x, migrant_r, migrant_stomach, migrant_models
        
        # Package migrants
        left_y, left_x, left_r, left_stomach, left_models = package_migrants(can_migrate_left, W)
        right_y, right_x, right_r, right_stomach, right_models = package_migrants(can_migrate_right, -W)
        
        # Exchange migrants with neighbors
        # Send left migrants to left neighbor
        left_payload_y = lax.cond(dev == 0, lambda _: jnp.full_like(left_y, -1), lambda _: left_y, operand=None)
        left_payload_x = lax.cond(dev == 0, lambda _: jnp.full_like(left_x, -1), lambda _: left_x, operand=None)
        left_payload_r = lax.cond(dev == 0, lambda _: jnp.zeros_like(left_r), lambda _: left_r, operand=None)
        left_payload_stomach = lax.cond(dev == 0, lambda _: jnp.zeros_like(left_stomach), lambda _: left_stomach, operand=None)
        left_payload_models = lax.cond(dev == 0, lambda _: jnp.zeros_like(left_models), lambda _: left_models, operand=None)
        
        incoming_right_y = lax.ppermute(left_payload_y, axis_name=AXIS, perm=perm_left_shift)
        incoming_right_x = lax.ppermute(left_payload_x, axis_name=AXIS, perm=perm_left_shift)
        incoming_right_r = lax.ppermute(left_payload_r, axis_name=AXIS, perm=perm_left_shift)
        incoming_right_stomach = lax.ppermute(left_payload_stomach, axis_name=AXIS, perm=perm_left_shift)
        incoming_right_models = lax.ppermute(left_payload_models, axis_name=AXIS, perm=perm_left_shift)
        
        # Send right migrants to right neighbor
        right_payload_y = lax.cond(dev == (ndev-1), lambda _: jnp.full_like(right_y, -1), lambda _: right_y, operand=None)
        right_payload_x = lax.cond(dev == (ndev-1), lambda _: jnp.full_like(right_x, -1), lambda _: right_x, operand=None)
        right_payload_r = lax.cond(dev == (ndev-1), lambda _: jnp.zeros_like(right_r), lambda _: right_r, operand=None)
        right_payload_stomach = lax.cond(dev == (ndev-1), lambda _: jnp.zeros_like(right_stomach), lambda _: right_stomach, operand=None)
        right_payload_models = lax.cond(dev == (ndev-1), lambda _: jnp.zeros_like(right_models), lambda _: right_models, operand=None)
        
        incoming_left_y = lax.ppermute(right_payload_y, axis_name=AXIS, perm=perm_right_shift)
        incoming_left_x = lax.ppermute(right_payload_x, axis_name=AXIS, perm=perm_right_shift)
        incoming_left_r = lax.ppermute(right_payload_r, axis_name=AXIS, perm=perm_right_shift)
        incoming_left_stomach = lax.ppermute(right_payload_stomach, axis_name=AXIS, perm=perm_right_shift)
        incoming_left_models = lax.ppermute(right_payload_models, axis_name=AXIS, perm=perm_right_shift)
        
        # Remove emigrants (bugs that successfully migrated)
        stay_mask = ~(can_migrate_left | can_migrate_right)
        new_active = active & stay_mask
        
        # Count emigrants for statistics
        num_emigrated_left = jnp.sum(can_migrate_left)
        num_emigrated_right = jnp.sum(can_migrate_right)
        
        # Clear emigrant data
        new_bug_x = jnp.where(stay_mask[:, None], bug_x, jnp.array([0, 0], dtype=jnp.int32))
        new_bug_r = jnp.where(stay_mask, bug_r, 0)
        new_bug_stomach = jnp.where(stay_mask, bug_stomach, 0.0)
        new_models = jnp.where(stay_mask[:, None], models, 0.0)
        
        # Place immigrants in their exact positions
        # Find valid incoming bugs
        valid_from_left = incoming_left_y >= 0
        valid_from_right = incoming_right_y >= 0
        
        # Find available slots
        available_slots = ~new_active
        
        # Super simplified immigrant placement - just place first few valid ones
        # This is fast and handles the common case where few bugs migrate
        def place_first_n_immigrants(incoming_y, incoming_x, incoming_r, incoming_stomach, incoming_models, valid_mask, n=32):
            """Place first N valid immigrants - fast implementation"""
            def place_one(i, carry):
                bug_x_c, bug_r_c, bug_stomach_c, models_c, active_c, slots_c = carry
                
                # Check if this is a valid immigrant and we have a slot
                is_valid = valid_mask[i]
                has_slot = jnp.sum(slots_c) > 0
                should_place = is_valid & has_slot
                
                # Find first available slot
                slot_idx = jnp.argmax(slots_c)
                
                # Place immigrant at their actual spatial position
                new_pos = jnp.array([incoming_y[i], incoming_x[i]], dtype=jnp.int32)
                bug_x_c = lax.cond(should_place,
                                 lambda x: x.at[slot_idx].set(new_pos),
                                 lambda x: x, bug_x_c)
                bug_r_c = lax.cond(should_place,
                                 lambda r: r.at[slot_idx].set(incoming_r[i]),
                                 lambda r: r, bug_r_c)
                bug_stomach_c = lax.cond(should_place,
                                       lambda s: s.at[slot_idx].set(incoming_stomach[i]),
                                       lambda s: s, bug_stomach_c)
                models_c = lax.cond(should_place,
                                  lambda m: m.at[slot_idx].set(incoming_models[i]),
                                  lambda m: m, models_c)
                active_c = lax.cond(should_place,
                                  lambda a: a.at[slot_idx].set(True),
                                  lambda a: a, active_c)
                slots_c = lax.cond(should_place,
                                 lambda s: s.at[slot_idx].set(False),
                                 lambda s: s, slots_c)
                
                return bug_x_c, bug_r_c, bug_stomach_c, models_c, active_c, slots_c
            
            carry = (new_bug_x, new_bug_r, new_bug_stomach, new_models, new_active, available_slots)
            # Only process first N positions to keep it fast
            carry = lax.fori_loop(0, min(n, max_players), place_one, carry)
            return carry
        
        # Place immigrants from left, then from right (process first 32 from each direction)
        carry = place_first_n_immigrants(
            incoming_left_y, incoming_left_x, incoming_left_r, 
            incoming_left_stomach, incoming_left_models, valid_from_left, n=32)
        new_bug_x, new_bug_r, new_bug_stomach, new_models, new_active, available_slots = carry
        
        carry = place_first_n_immigrants(
            incoming_right_y, incoming_right_x, incoming_right_r,
            incoming_right_stomach, incoming_right_models, valid_from_right, n=32)
        new_bug_x, new_bug_r, new_bug_stomach, new_models, new_active, _ = carry
        
        # Rebuild bug_object_grid
        world_with_halo = (H, W + 2 * halo)
        new_bug_object_grid = jnp.full(world_with_halo, -1, dtype=jnp.int32)
        new_bug_object_grid = new_bug_object_grid.at[new_bug_x[...,0], new_bug_x[...,1]].set(
            jnp.where(new_active, jnp.arange(max_players), -1))
        
        # Update migration statistics
        new_stats = stats.at[dev, 0].add(num_emigrated_left)  # Left emigrations
        new_stats = new_stats.at[dev, 1].add(num_emigrated_right)  # Right emigrations
        
        return new_bug_x, new_bug_r, new_bug_stomach, new_models, new_active, new_bug_object_grid, new_stats
    
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
        bug_object_halo_copy = make_bug_object_halo_copy(H, W, halo, ndev)
        bug_migration = make_bug_migration_system(H, W, halo, ndev, params.max_players, bug_object_halo_copy)
        
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
            
            # Update state (keep migrated as-is, migration function will update it)
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
            
            # Use scatter-add for rendering bugs (ensure correct dtypes)
            bug_indices = jnp.where(valid_positions[:, None], interior_bug_x, jnp.array([H-1, W-1], dtype=jnp.int32))
            bug_colors_to_add = jnp.where(valid_positions[:, None], bug_color * 0.5, 0.0).astype(jnp.float32)
            image = image.at[bug_indices[:, 0], bug_indices[:, 1]].add(bug_colors_to_add)
            
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
    parser.add_argument('--profile_communication', action='store_true', help='Profile GPU communication performance')
    parser.add_argument('--verify_gpu_direct', action='store_true', default=True, help='Verify direct GPU communication')
    return parser.parse_args()

def diagnose_gpu_communication():
    """Diagnose GPU communication capabilities and performance"""
    print("\n" + "="*60)
    print("GPU COMMUNICATION DIAGNOSTICS")
    print("="*60)
    
    # 1. Check JAX backend
    print(f"\n1. JAX Backend: {jax.default_backend()}")
    print(f"   JAX version: {jax.__version__}")
    
    # 2. Device information
    devices = jax.devices()
    print(f"\n2. Devices ({len(devices)} total):")
    for i, dev in enumerate(devices):
        print(f"   [{i}] {dev}")
        print(f"       Platform: {dev.platform}")
        print(f"       Device kind: {dev.device_kind}")
    
    # 3. Check for GPU peer-to-peer access (NVIDIA specific)
    if jax.default_backend() == 'gpu':
        print("\n3. GPU Communication Check:")
        try:
            import subprocess
            # Check NCCL availability
            result = subprocess.run(['nvidia-smi', 'topo', '-m'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("   GPU Topology (nvidia-smi topo -m):")
                print("   " + "\n   ".join(result.stdout.split('\n')[:20]))
            
            # Check for NVLink (per-GPU, check for bandwidth values)
            print("\n   NVLink Status (per GPU):")
            nvlink_active = False
            for gpu_id in range(len(devices)):
                result = subprocess.run(['nvidia-smi', 'nvlink', '--status', '-i', str(gpu_id)],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Check for bandwidth values (e.g., "26.562 GB/s" means link is active)
                    if 'GB/s' in result.stdout:
                        nvlink_active = True
                        active_count = result.stdout.count('GB/s')
                        print(f"   GPU {gpu_id}: {active_count} active NVLink(s)")
                    elif 'Active' in result.stdout:
                        nvlink_active = True
                        active_count = result.stdout.count('Active')
                        print(f"   GPU {gpu_id}: {active_count} active NVLink(s)")
            
            if nvlink_active:
                print("   NVLink connections active (GPU-direct enabled)")
            else:
                print("   No active NVLink connections (may use PCIe/CPU)")
        except Exception as e:
            print(f"   Could not check GPU topology: {e}")
    
    # 4. JAX distributed configuration
    print("\n4. JAX Distributed Config:")
    print(f"   XLA flags: {os.environ.get('XLA_FLAGS', 'Not set')}")
    print(f"   NCCL debug: {os.environ.get('NCCL_DEBUG', 'Not set')}")
    print(f"   XLA backend: {os.environ.get('JAX_PLATFORMS', jax.default_backend())}")
    
    # 5. Communication backend test
    print("\n5. Testing collective operations:")
    try:
        ndev = min(4, len(devices))
        test_data = jnp.arange(ndev * 100).reshape(ndev, 100)
        
        def test_allreduce(x):
            return lax.psum(x, axis_name='batch')
        
        test_allreduce_pm = pmap(test_allreduce, axis_name='batch', devices=devices[:ndev])
        
        import time
        start = time.time()
        result = test_allreduce_pm(test_data)
        result.block_until_ready()
        elapsed = time.time() - start
        
        # Calculate bandwidth
        data_size = test_data.nbytes * 2  # all-reduce: send + receive
        bandwidth_gbps = (data_size / (elapsed if elapsed > 0 else 1e-6)) / 1e9
        print(f"   Collective operations: {elapsed*1000:.2f}ms, {bandwidth_gbps:.2f} GB/s")
        
        # Test ppermute
        def test_ppermute(x):
            perm = [(i, (i+1) % ndev) for i in range(ndev)]
            return lax.ppermute(x, 'batch', perm)
        
        test_ppermute_pm = pmap(test_ppermute, axis_name='batch', devices=devices[:ndev])
        
        start = time.time()
        result = test_ppermute_pm(test_data)
        result.block_until_ready()
        elapsed = time.time() - start
        
        bandwidth_gbps = (test_data.nbytes / (elapsed if elapsed > 0 else 1e-6)) / 1e9
        print(f"   Point-to-point (ppermute): {elapsed*1000:.2f}ms, {bandwidth_gbps:.2f} GB/s")
        
    except Exception as e:
        print(f"   Communication test failed: {e}")
    
    print("\n" + "="*60)
    
    print("="*60 + "\n")

def benchmark_communication(ndev: int, H: int, W: int, halo: int):
    """Benchmark actual communication bandwidth for halo exchange"""
    print("\n" + "="*60)
    print("COMMUNICATION BANDWIDTH BENCHMARK")
    print("="*60)
    
    devices = jax.devices()[:ndev]
    
    # Create test food grid
    food_grid_shape = (H, W + 2*halo)
    test_grids = jnp.ones((ndev, *food_grid_shape), dtype=jnp.float32)
    
    # Benchmark halo exchange
    from dirt.envs.simple_env_dist import make_food_halo_copy
    halo_copy = make_food_halo_copy(H, W, halo, ndev)
    halo_copy_pm = pmap(halo_copy, in_axes=(0, 0), axis_name='mesh', devices=devices)
    
    roles = jnp.arange(ndev, dtype=jnp.int32)
    
    import time
    
    # Warmup
    for _ in range(5):
        _ = halo_copy_pm(test_grids, roles)
    
    # Benchmark
    n_trials = 50
    times = []
    for _ in range(n_trials):
        start = time.time()
        result = halo_copy_pm(test_grids, roles)
        result.block_until_ready()
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000
    
    # Calculate bandwidth
    bytes_per_halo = halo * H * 4  # float32 = 4 bytes
    bytes_transferred = 2 * bytes_per_halo * ndev  # 2 halos per device
    bandwidth_gbps = (bytes_transferred / (avg_time / 1000)) / 1e9
    
    print(f"\nHalo Exchange Performance:")
    print(f"  Grid shape per device: {food_grid_shape}")
    print(f"  Halo width: {halo}")
    print(f"  Bytes per halo: {bytes_per_halo:,}")
    print(f"  Total bytes transferred: {bytes_transferred:,}")
    print(f"  Average time: {avg_time:.3f} ± {std_time:.3f} ms")
    print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
    
    # Reference bandwidths
    print(f"\nReference Bandwidths:")
    print(f"  NVLink 3.0: ~300 GB/s (per link)")
    print(f"  NVLink 2.0: ~150 GB/s (per link)")
    print(f"  PCIe 4.0 x16: ~32 GB/s")
    print(f"  PCIe 3.0 x16: ~16 GB/s")
    print(f"  CPU memory: ~1-10 GB/s (bottleneck!)")
    
    if bandwidth_gbps < 10:
        print(f"\n⚠ Low ppermute bandwidth (expected for JAX pmap)")
        print(f"  JAX ppermute has high overhead and limited bandwidth utilization.")
        print(f"  This is a known limitation, not a hardware/NCCL issue.")
        print(f"  For small halos (3 pixels), overhead is negligible vs computation.")
        print(f"\n  To verify NCCL/NVLink is working, check collective ops bandwidth:")
        print(f"  - Run the full diagnostic to see all-reduce performance")
        print(f"  - Collective ops should show >50 GB/s if NVLink is active")
    elif bandwidth_gbps > 50:
        print(f"\nExcellent bandwidth - using NVLink or similar high-speed interconnect")
    
    print("="*60 + "\n")

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
        profile_communication=args.profile_communication,
        verify_gpu_direct=args.verify_gpu_direct,
    )
    
    H, W = params.world_size
    ndev = params.ndev
    
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.device_count()}")
    print(f"Device list: {jax.devices()}")
    
    # Run diagnostics if requested
    if params.verify_gpu_direct:
        diagnose_gpu_communication()
    
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
    
    # Benchmark communication if requested
    if params.profile_communication:
        benchmark_communication(ndev, H, W, params.halo)
    
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
    video_frames = []
    if params.save_video:
        # Calculate sampling frequency for desired video length
        target_frames = min(params.video_fps * params.video_length, params.max_video_frames)
        video_sample_freq = max(1, params.steps // target_frames)
        print(f"Video sampling: every {video_sample_freq} steps for {target_frames} frames ({target_frames/params.video_fps:.1f}s at {params.video_fps}fps)")
    else:
        video_sample_freq = params.steps + 1  # Never render
    
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
    
    # Create a combined step function to reduce pmap overhead
    def combined_step(keys, state, models, obs, role, step_num):
        """Combined step: act -> transition -> observe -> update_models"""
        action_key, env_key, obs_key, mutate_key = jrng.split(keys, 4)
        
        # Get actions
        u = jrng.uniform(action_key, models.shape, minval=1e-6, maxval=1.)
        gumbel = -jnp.log(-jnp.log(u))
        actions = jnp.argmax(models + gumbel, axis=-1)
        
        # Environment step
        next_state = transition_fn(env_key, state, actions, None, role)
        
        # Get next observations
        next_obs = observe_fn(obs_key, next_state)
        
        # Update models based on reproduction
        parents, children = family_fn(next_state)
        parent_models = models[parents]
        mutated_parents = mutate(mutate_key, parent_models)
        next_models = models.at[children].set(mutated_parents[..., 0])
        
        return next_state, next_models, next_obs
    
    combined_step_pm = pmap(combined_step, in_axes=(0, 0, 0, 0, 0, None), axis_name=AXIS, devices=devices)
    
    # Define migration step function OUTSIDE the loop
    if params.allow_migration:
        def migrate_step(state, models, stats, role):
            active = active_fn(state)
            new_bug_x, new_bug_r, new_bug_stomach, new_models, new_active, new_bug_object_grid, new_stats = migrate_fn(
                state.bug_x, state.bug_r, state.bug_stomach, active, models, state.bug_object_grid, stats, role)
            
            # Update state with migrated bugs
            new_state = state.replace(
                bug_x=new_bug_x,
                bug_r=new_bug_r, 
                bug_stomach=new_bug_stomach,
                bug_object_grid=new_bug_object_grid
            )
            return new_state, new_models, new_stats
        
        migrate_pm = pmap(migrate_step, in_axes=(0, 0, 0, 0), axis_name=AXIS, devices=devices)
    
    print(f"Starting main simulation loop...")
    print(f"TIP: Use Ctrl+C to stop early and still save video")
    
    # Initialize migration statistics
    if params.allow_migration:
        sharded_migration_stats = jnp.zeros((ndev, ndev, 2), dtype=jnp.int32)
    
    # Run simulation with evolution
    for step in range(params.steps):
        # Use combined step for better GPU utilization
        key, step_key = jrng.split(key)
        step_keys = jrng.split(step_key, ndev)
        
        sharded_state, sharded_models, sharded_obs = combined_step_pm(
            step_keys, sharded_state, sharded_models, sharded_obs, roles, step)
        
        # Bug migration step (if enabled and at right frequency)
        if params.allow_migration and (step % params.migration_freq == 0):
            sharded_state, sharded_models, sharded_migration_stats = migrate_pm(
                sharded_state, sharded_models, sharded_migration_stats, roles)
        
        # Render for video (keep on GPU until end to avoid blocking)
        if params.save_video and step % video_sample_freq == 0:
            sharded_renders = render_pm(sharded_state)
            video_frames.append(sharded_renders)  # Keep on GPU
        
        # Progress logging
        if (step + 1) % max(1, params.steps // 10) == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"Step {step + 1}/{params.steps} ({steps_per_sec:.1f} steps/sec)")
    
    print("Simulation complete!")
    
    # Print migration statistics
    if params.allow_migration:
        print("\n" + "="*60)
        print("MIGRATION REPORT")
        print("="*60)
        stats_cpu = np.array(sharded_migration_stats)
        total_migrations = 0
        for dev in range(ndev):
            left_emigrants = int(stats_cpu[dev, dev, 0])
            right_emigrants = int(stats_cpu[dev, dev, 1])
            dev_total = left_emigrants + right_emigrants
            total_migrations += dev_total
            
            if dev_total > 0:
                left_target = (dev - 1) if dev > 0 else "boundary"
                right_target = (dev + 1) if dev < ndev - 1 else "boundary"
                print(f"\nGPU {dev}:")
                if left_emigrants > 0:
                    print(f"  Left to GPU {left_target}: {left_emigrants} bugs")
                if right_emigrants > 0:
                    print(f"  Right to GPU {right_target}: {right_emigrants} bugs")
        
        print(f"\nTotal migrations: {total_migrations} bugs")
        if params.steps > 0:
            avg_per_step = total_migrations / params.steps
            print(f"Average per step: {avg_per_step:.2f} bugs")
        print("="*60 + "\n")
    
    # Save video
    if params.save_video and video_frames:
        import imageio
        print(f"Transferring {len(video_frames)} video frames from GPU...")
        # Transfer all frames from GPU to CPU at once
        cpu_frames = [stitch_global_image(frame, ndev) for frame in video_frames]
        
        print(f"Saving video to {params.video_path}...")
        video_frames = np.array(cpu_frames)
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