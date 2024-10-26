def leapfrog_scalar_step(x0, dx0, f, dt=1., mass=1., drag=0.):
    # first half of the acceleration with drag
    fm = f/mass
    dm = drag/mass
    ddx0 = fm - dm * dx0
    
    # first half of the velocity update
    dxh = dx0 + 0.5 * dt * ddx0
    
    # position update
    x1 = x0 + dt * dxh
    
    # second half of the acceleration with drag
    ddx1 = fm - dm * dxh
    
    # second half of the velocity update
    dx1 = dxh + 0.5 * dt * ddx1
    
    # return
    return x1, dx1
