import jax.numpy as jnp

from dirt.dynamics import gridworld2d as g2d

def extract_fpv(x, r, observable_area, observation_map, out_of_bounds=0):
    
    # compute the height, width and sign of the fpv
    #h,w = b-a
    a, b = observable_area
    s = jnp.sign(b-a)
    
    # construct local rows and columns that describe the area around the agent
    # that should be extracted based on each of the four rotation directions
    '''
    rows = jnp.zeros((4,h,w), dtype=jnp.int32)
    rows = rows.at[0,:,:].set(jnp.arange(a[0], b[0], s[0])[:,None])
    rows = rows.at[1,:,:].set(jnp.arange(a[1], b[1], s[1])[None,:])
    rows = rows.at[2,:,:].set(jnp.arange(b[0]-1, a[0]-1, -s[0])[:,None])
    rows = rows.at[3,:,:].set(jnp.arange(b[1]-1, a[1]-1, -s[0])[None,:])
    cols = rows[jnp.array([1,2,3,0])]
    
    # compute global rows and columns by pulling out the local
    # rows and columns corresponding to the rotation r of
    # each agent then add the position x of each agent
    xr_rows = rows[r] + x[...,0,None,None]
    xr_cols = cols[r] + x[...,1,None,None]
    '''
    #rc = jnp.zeros((h,w,2), dtype=jnp.int32)
    #rc = rc.at[:,:,0].set(jnp.arange(a[0], b[0], s[0])[:,None])
    #rc = rc.at[:,:,1].set(jnp.arange(a[1], b[1], s[1])[None,:])
    
    # construct the local rows and columns that describe the area around the
    # agents that should be extracted
    # (h, w)
    local_rc = jnp.stack(jnp.meshgrid(
        jnp.arange(a[0], b[0], s[0]),
        jnp.arange(a[1], b[1], s[1]),
        indexing='ij',
    ), axis=-1)
    
    # construct the global rows and columns by rotating and offsetting the
    # local rows and columns using the agents' positions and orientations
    global_rc = g2d.rotate(local_rc, r[...,None, None]) + x[...,None,None,:]
    
    # cut out regions of the observation map corresponding to the computed
    # rows and columns
    fpv = observation_map.at[global_rc[...,0], global_rc[...,1]].get(
        mode='fill', fill_value=out_of_bounds)
    
    # return
    return fpv

if __name__ == '__main__':
    observation_map = jnp.arange(5*5).reshape(5,5)
    x = jnp.array([
        [2,2],
        [2,2],
        [2,2],
        [2,2],
    ])
    r = jnp.array([
        0,
        1,
        2,
        3,
    ])
    a = jnp.array([-2,-2])
    b = jnp.array([3,3])
    obs = extract_fpv(x, r, a, b, observation_map)
    
    direction0 = g2d.rotate(jnp.array([1,0]), 0)
    direction1 = g2d.rotate(jnp.array([1,0]), 1)
    direction2 = g2d.rotate(jnp.array([1,0]), 2)
    direction3 = g2d.rotate(jnp.array([1,0]), 3)
    
    breakpoint()
