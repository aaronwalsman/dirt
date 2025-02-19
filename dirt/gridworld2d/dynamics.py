from typing import Tuple, Optional, Union

import jax.numpy as jnp

'''
gridworld2d dynamics:

Rigid objects have a position (x) and orientation (r)
and are acted upon by a translation (dx) and a rotation (dr).

Positions (x) are 2d integers between (0,0) and (h,w).
Translations (dx) are unbounded 2d integers.
Orientations (r) are integers between 0 and 4.
Rotations (dr) are unbounded integers.

Dynamics expressed in global coordinates are computed as:
x1 = (clip or wrap)(x0 + dx)
r1 = (wrap)(r0 + dr)

Dynamics expressed in local coordinates are computed as:
x1 = (clip or wrap)(x0 + r0 * dx)
r1 = (wrap)(r0 + dr)
'''
gridworld2d_rotation_matrices = jnp.array([
    [[ 1, 0], [ 0, 1]],
    [[ 0,-1], [ 1, 0]],
    [[-1, 0], [ 0,-1]],
    [[ 0, 1], [-1, 0]],
])

inverse_rotations = jnp.array([0,3,2,1])

def rotate(
    x : jnp.ndarray,
    r : jnp.ndarray,
    pivot : jnp.ndarray | int = 0,
) -> jnp.ndarray :
    '''
    Rotates a position or direction x by a discrete rotation r about a pivot.
    
    x : The position/direction to rotate.
    r : The rotation ammount.
    pivot : The rotation pivot point.
    '''
    m = gridworld2d_rotation_matrices[r]
    # the [...,None] forces jnp to recognize x as a 2x1-vector
    # and [...,0] clips it back to a 2-vector again
    return (m @ (x-pivot)[...,None])[...,0] + pivot

def wrap_x(
    x : jnp.ndarray,
    world_size : Tuple[int, int],
) -> jnp.ndarray :
    '''
    Wraps a position x around the borders of a gridworld with a torus topology.
    
    x : The position to wrap.
    world_size : The size of the grid to determine the wrap borders.
    '''
    return x % world_size

def clip_x(
    x : jnp.ndarray,
    world_size : Tuple[int, int],
) -> jnp.ndarray :
    '''
    Clips a position x to stay within the boundaries of the gridworld.
    
    x : The position to clip.
    world_size : The size of the grid to determine the clip borders.
    '''
    return jnp.clip(x, jnp.array([0,0]), jnp.array(world_size))

def wrap_r(
    r : jnp.ndarray,
) -> jnp.ndarray :
    '''
    Remaps discrete orientations to lie between 0 and 3.
    
    r : The rotations to wrap.
    '''
    return r % 4

def move_mass(
    x0 : jnp.ndarray,
    x1 : jnp.ndarray,
    mass_grid : jnp.ndarray,
    mass : Union[int, float, jnp.ndarray] = 1,
) -> jnp.ndarray :
    '''
    Returns a new mass_grid after a fixed ammount of mass at one position x0
    has been moved to another position x1.
    
    x0 : The starting position for the transition.  Mass will be
        subtracted from these locations.
    x1 : The ending position for the transition.  Mass will be
        added to these locations.
    occupancy : The existing occupancy map.
    '''
    mass_grid = mass_grid.at[x0[...,0], x0[...,1]].add(-mass)
    mass_grid = mass_grid.at[x1[...,0], x1[...,1]].add(mass)
    return mass_grid

def collide(
    x0 : jnp.ndarray,
    x1 : jnp.ndarray,
    occupancy_grid : jnp.ndarray,
    max_occupancy : int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] :
    '''
    Checks for collisions as objects at x0 move to x1.  Objects are
    considered to have collided if they land on a location that another object
    moved either from or to.  The movement of any colliding objects is undone.
    
    Updates the occupancy_grid based on the movement from x0 to x1, then
    checks if any of the updates have resulted in too many items in the
    same position and undoes all movements where this has occurred.
    Returns a modified version of x1 where all colliding positions have
    been reset to x0, a vector representing which items collided, and an
    updated occupancy map.
    
    x0 : The starting position for each moving object.
    x1 : The ending position for each moving object.
    occupancy_grid : The occupancy grid before the motion.
    '''
    collision_grid = occupancy_grid | move_mass(x0, x1, occupancy_grid)
    collided = (collision_grid[x1[:,0], x1[:,1]] > 1)[...,None]
    x2 = jnp.where(collided, x0, x1) #x1 * ~collided + x0 * collided
    occupancy_grid = move_mass(x0, x2, occupancy_grid)
    return x2, collided, occupancy_grid

def move_objects(
    x0 : jnp.ndarray,
    x1 : jnp.ndarray,
    object_grid : jnp.ndarray,
    empty : int = -1,
) -> jnp.ndarray :
    '''
    Returns a new object_grid after objects at position x0
    have been moved to another position x1.  This does not do any
    collision checking (see move_and_collide_objects below), so if
    two objects are moved to the same locations, one will be overwritten.
    
    x0 : The starting position for the transition.  Objects will be
        removed from these locations.
    x1 : The ending position for the transition.  Objects will be
        placed in these locations.
    object_grid : The existing object_grid.
    '''
    values = object_grid[x0[...,0], x0[...,1]]
    object_grid = object_grid.at[x0[...,0], x0[...,1]].set(empty)
    object_grid = object_grid.at[x1[...,0], x1[...,1]].set(values)
    return object_grid

def move_and_collide_objects(
    x0 : jnp.ndarray,
    x1 : jnp.ndarray,
    object_grid : jnp.ndarray,
    empty : int = -1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] :
    '''
    Returns a new object_grid after attempting to move objects at
    position x0 has been moved to another position x1
    
    x0 : The starting position for the transition.  Mass will be
        subtracted from these locations.
    x1 : The ending position for the transition.  Mass will be
        added to these locations.
    object_grid : The existing object_grid.
    '''
    occupancy_grid = (object_grid != empty).astype(jnp.int32)
    x2, collided, _ = collide(x0, x1, occupancy_grid)
    object_grid = move_objects(x0, x2, object_grid, empty=empty)
    return x2, collided, object_grid

def step(
    x0 : jnp.ndarray,
    r0 : jnp.ndarray,
    dx : jnp.ndarray,
    dr : jnp.ndarray,
    space : str = 'global',
    world_size : Optional[Tuple[int, int]] = None,
    check_collisions : bool = False,
    object_grid : Optional[jnp.ndarray] = None,
    out_of_bounds : str = 'clip',
) -> Tuple[jnp.ndarray, jnp.ndarray] :
    '''
    Applies gridworld2d dynamics to a set of positions x0 and orientations r0
    based on velocities dx and angular velocities dr.  Offsets can be
    applied in either global or local space.  If check_collisions is True, an
    object_grid must be specified.  If check_collisions is False and an
    object_grid is specified, the object_grid is updated without
    checking for collisions.
    
    x0 : The current position of the agents/objects.
    r0 : The current orientation of the agents/objects.
    dx : The velocity of the agents/objects.
    dr : The angular velocity of the agents/objects.
    space : Should be "global" or "local" and specifies whether the velocity
        is specified in the local coordinate frame of each agent/object.
    world_size : The size of the gridworld.  Used for clipping or wrapping the
        positions if they move out of bounds.  If omitted, this will be inferred
        from the object_grid grid if that is specified.  If neither are
        specified, out_of_bounds must be "none" indicating no border checks
        will be enforced.
    check_collisions: Whether or not to check if two objects will collide if
        they move to the same location.
    object_grid: The object_grid to use for collision checking.  If
        specified, this will be updated based on the motion of the objects.
        If specified, but check_collisions is False, this will still be updated
        based on the motion of the agents.
    max_occpuancy : The maximum number of objects or agents that can occupy a
        single grid cell.
    out_of_bounds : How to handle out_of_bounds behavior if the agent leaves
        the edge of the gridworld.  Can be either "clip", "wrap" or "none".
    '''
    if world_size is None and object_grid is not None:
        world_size = object_grid.shape
    
    if space == 'global':
        x1 = x0 + dx
    elif space == 'local':
        x1 = x0 + rotate(dx, r0)
    else:
        raise NotImplementedError
    
    if out_of_bounds == 'clip':
        assert world_size is not None
        x1 = clip_x(x1, world_size)
    elif out_of_bounds == 'wrap':
        assert world_size is not None
        x1 = wrap_x(x1, world_size)
    elif out_of_bounds == 'none':
        pass
    
    r1 = wrap_r(r0 + dr)
    
    if object_grid is not None:
        if check_collisions:
            x1, collided, object_grid = move_and_collide_objects(
                x0, x1, object_grid)
            return x1, r1, collided, object_grid
        
        else:
            object_grid = update_occupancy(x0, x1, object_grid)
            return x1, r1, object_grid
    
    return x1, r1

def forward_rotate_step(
    x0 : jnp.ndarray,
    r0 : jnp.ndarray,
    forward : jnp.ndarray,
    rotate : jnp.ndarray,
    **kwargs
) -> Tuple[jnp.ndarray, jnp.ndarray] :
    dx = jnp.stack((forward, jnp.zeros_like(forward)), axis=-1)
    return step(x0, r0, dx, rotate, space='local', **kwargs)

if __name__ == '__main__':
    x0 = jnp.array([
        [ 0, 1],
        [ 1, 0],
        [ 3, 3],
        [ 0, 3],
    ])
    x1 = jnp.array([
        [ 1, 1],
        [ 1, 1],
        [ 2, 3],
        [ 0, 2],
    ])
    occupancy = jnp.array([
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ])
    
    x2, occupancy = collide(x0, x1, occupancy)
