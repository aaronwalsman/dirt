import jax.numpy as jnp
import jax.random as jrng
import jax

from dirt.motion.leapfrog import leapfrog_scalar_step

'''
scalar2d dynamics

Rigid objects have a position (x), a velocity (dx), an orientation (r) and
an angular velocity (dr).  They are acted upon using an acceleration (ddx) and
an angular acceleration (ddr).

positions (x) are 2d floats between (0,0) and (w,h)
translations (dx) are unbounded 2d floats
accelerations (ddx) are unbounded 2d floats
orientations (r) are floats between 0 and 2pi
angular velocity (dr) are unbounded floats
angular acceleration (ddr) are unbounded floats

dynamics are computed as:
x1a, dx1 = (integrate)(x0, dx0, r0 * ddx)
x1 = (clip or wrap)(x1a)
r1a, dr1 = (integrate)(r0, dr0, ddr)
r1 = (wrap)(r1a)
'''

def rotation_matrix(r):
    '''
    computes a 2D rotation matrix from a scalar angle r
    '''
    return jnp.array([
        [jnp.cos(r), -jnp.sin(r)],
        [jnp.sin(r),  jnp.cos(r)],
    ])

def rotate(x, r, pivot=0.):
    '''
    rotates a position, direction or acceleration x by an angle r about a pivot
    '''
    m = rotation_matrix(r)
    return (m @ (x-pivot)[...,None])[...,0] + pivot

def wrap_x(x, world_size):
    '''
    wraps a position x around the borders of a gridworld with a torus topology
    '''
    return jnp.fmod(x, world_size)

def clip_x(x, world_size):
    '''
    clips a position x to stay within the boundaries of the world
    '''
    return jnp.clip(x, [0,0], world_size)

def wrap_r(r):
    '''
    remaps an angle r to lie between 0 and 2pi
    '''
    return jnp.fmpd(r, 0., jnp.pi*2.)

def step_wrap(x0, dx0, ddx, r0, dr0, ddr, dt=1., drag=1.):
    '''
    computes a new position x1, velocity dx1, orientation r1 and angular
    velocity dr1 given a previous position x0, velocity dx0, acceleration ddx,
    orientation r0, angular velocity dr0 and angular acceleration ddr
    wraps x1 to stay within the world size (torus topology)
    '''
    ddx_global = rotate(ddx, r0)
    x1, dx1 = leapfrog_scalar_step(x0, dx0, ddx_global, dt, drag=drag)
    x1 = wrap_x(x1)
    r1, dr1 = leapfrog_scalar_step(r0, dr0, ddr, dt)
    r1 = wrap_r(r1)
    return x1, dx1, r1, dr1

def step_clip(x0, dx0, ddx, r0, dr0, ddr, dt=1., drag=1.):
    '''
    computes a new position x1, velocity dx1, orientation r1 and angular
    velocity dr1 given a previous position x0, velocity dx0, acceleration ddx,
    orientation r0, angular velocity dr0 and angular acceleration ddr
    clips x1 to stay within the world size (torus topology)
    '''
    ddx_global = rotate(ddx, r0)
    x1, dx1 = leapfrog_scalar_step(x0, dx0, ddx_global, dt, drag=drag)
    x1 = clip_x(x1)
    r1, dr1 = leapfrog_scalar_step(r0, dr0, ddr, dt)
    r1 = wrap_r(r1)
    return x1, dx1, r1, dr1
