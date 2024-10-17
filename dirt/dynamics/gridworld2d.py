import jax.numpy as jnp

'''
gridworld2d dynamics

rigid objects have a position (x) and orientation (r)
and are acted upon by a translation (dx) and a rotation (dr)

positions (x) are 2d integers between (0,0) and (h,w)
translations (dx) are unbounded 2d integers
orientations (r) are integers between 0 and 4
rotations (dr) are unbounded integers

dynamics are computed as:
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

def rotate(x, r, pivot=0):
    '''
    rotates a position or direction x by a discrete rotation r about a pivot
    '''
    m = gridworld2d_rotation_matrices[r]
    return (m @ (x-pivot)[...,None])[...,0] + pivot

def wrap_x(x, world_size):
    '''
    wraps a position x around the borders of a gridworld with a torus topology
    '''
    return x % world_size

def clip_x(x, world_size):
    '''
    clips a position x to stay within the boundaries of the gridworld
    '''
    return jnp.clip(x, jnp.array([0,0]), world_size)

def wrap_r(r):
    '''
    remaps discrete orientations to lie between 0 and 3
    '''
    return r % 4

def step_wrap(x0, dx, r0, dr, world_size):
    '''
    computes a new position x1 and orientation r1 given a previous
    position x0, orientation r0 along with a translation dx and rotation dr
    wraps x1 to stay within the world_size (torus topology)
    '''
    x1 = wrap_x(x0 + rotate(dx, r0), world_size)
    r1 = wrap_r(r0 + dr)
    return x1, r1

def step_clip(x0, dx, r0, dr, world_size):
    '''
    computes a new position x1 and orientation r1 given a previous
    position x0, orientation r0 along with a translation dx and rotation dr
    clips x1 to stay within the world_size
    '''
    x1 = clip_x(x0 + rotate(dx, r0), world_size)
    r1 = wrap_r(r0 + dr)
    return x1, r1

def update_occupancy(x0, x1, occupancy):
    '''
    subtracts one from all locations at x0 and adds one to all locations at x1
    '''
    occupancy = occupancy.at[x0[:,0], x0[:,1]].add(-1)
    occupancy = occupancy.at[x1[:,0], x1[:,1]].add(1)
    return occupancy

def collide(x0, x1, occupancy, max_occupancy=1):
    '''
    updates the occupancy based on the movement from x0 to x1, then checks if
    any of the updates have resulted in too many items in the same location and
    undos all movements where this has occurred
    '''
    occupancy = update_occupancy(x0, x1, occupancy)
    collided = (occupancy[x1[:,0], x1[:,1]] > 1)[...,None]
    #x2 = x1.at[collided].set(x0[collided])
    x2 = x1 * ~collided + x0 * collided
    occupancy = update_occupancy(x1, x2, occupancy)
    return x2, collided, occupancy

def step_wrap_collide(x0, dx, r0, dr, occupancy, max_occupancy=1):
    '''
    applies a step with wrapping, then checks for collision and undos any
    colliding actions
    '''
    world_size = occupancy.shape
    x1, r1 = step_wrap(x0, dx, r0, dr, world_size)
    x2, collided, occupancy = collide(
        x0, x1, occupancy, max_occpuancy=max_occupancy)
    return x2, r1, collided, occupancy

def step_clip_collide(x0, dx, r0, dr, occupancy, max_occupancy=1):
    '''
    applies a step with clipping, then checks for collision and undos any
    colliding actions
    '''
    world_size = jnp.array(occupancy.shape)
    x1, r1 = step_clip(x0, dx, r0, dr, world_size)
    x2, collided, occupancy = collide(
        x0, x1, occupancy, max_occupancy=max_occupancy)
    return x2, r1, collided, occupancy

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
