import jax
import jax.numpy as jnp

'''
Designed to allow operations over grids of multiple sizes that represent
float-based quantities distributed over the same fixed area.  For example a
grid with shape (4x4) and another with shape (64x64) may represent different
levels of granularity over the same 1 km^2.  Quantities at coarser levels of
detail are summed versions of their higher resolution counterparts.  This means
that downsampling and upsampling a grid should maintain the overall sum of
quantities across the entire grid.

This module does not do any interpolation, and so requires grid shapes to be
evenly divisible by each other in order to be compatible.
'''

def grid_to_blocks(a, h, w):
    '''
    Reshapes a grid of size (ah, aw, *c) to blocks of size
    (h, ah//h, w, aw//w, *c).
    '''
    ah, aw, *c = a.shape
    assert ah % h == 0 and aw % w == 0
    return a.reshape(h, ah//h, w, aw//w, *c)

def blocks_to_grid(a):
    '''
    Reshapes blocks of size (ah, dh, aw, dw, *c) to (ah*dh, aw*dw, *c).
    '''
    h, dh, w, dw, *c = a.shape
    return a.reshape(h*dh, w*dw, *c)

def downsample_grid(a, h, w):
    '''
    Reduces the size of a grid by summing local blocks of values.
    '''
    a = grid_to_blocks(a, h, w)
    return jnp.sum(a, axis=(1,3))

def upsample_grid(a, h, w):
    '''
    Increases the size of a grid by repeating and dividing coarse values.
    
    Note: this function is included for the sake of compleness, but should be
    avoided in favor of broadcasting in most cases.
    '''
    ah, aw, *c = a.shape
    assert h >= ah and w >= aw
    dh = h // ah
    dw = w // aw
    block = jnp.full((dh, dw), 1./(dh*dw))
    return jnp.kron(a, block)

def grids_to_aligned_blocks(a, b):
    '''
    Returns the blocked version of two grids such that they share the largest
    common height and width so that they are broadcast compatible.  If the
    resolution of a is larger than b, this will produce grids with size:
    a: (h, dh, w, dw, *c)
    b: (h,  1, w,  1, *c)
    Otherwise:
    a: (h,  1, w,  1, *c)
    b: (h, dh, w, dw, *c)
    '''
    ah, aw, *ac = a.shape
    bh, bw, *bc = b.shape
    min_h = min(ah, bh)
    min_w = min(aw, bw)
    return grid_to_blocks(a, min_h, min_w), grid_to_blocks(b, min_h, min_w)

def scale_grid(a, alpha):
    a, alpha = grids_to_aligned_blocks(a, alpha)
    dah, daw = a.shape[1], a.shape[3]
    scale_factor = 1. / (alpha.shape[1] * alpha.shape[3])
    a = a * scale_factor * alpha
    if dah == 1:
        a = jnp.sum(a, axis=1, keepdims=True)
    if daw == 1:
        a = jnp.sum(a, axis=3, keepdims=True)
    a = blocks_to_grid(a)
    return a

def interpolate_grids(a, b, alpha, preserve_mass=True):
    a = scale_grid(a, alpha)
    b = scale_grid(b, 1. - alpha)
    return add_grids(a, b, preserve_mass=preserve_mass)

def add_grids(a, b, preserve_mass=True):
    '''
    Adds two grids with different shapes, the result will be the same size as
    the first grid.
    '''
    a, b = grids_to_aligned_blocks(a, b)
    if preserve_mass:
        scale_factor = a.shape[1] * a.shape[3]
    else:
        scale_factor = 1.
    b = jnp.sum(b, axis=(1,3), keepdims=True) / scale_factor
    c = a + b
    c = blocks_to_grid(c)
    return c

def take_grids(a, b, eps=1e-8):
    '''
    Subtract min(a,b) from a. Returns the remaining quantity a - min(a,b)
    and the taken quantity min(a,b).  The remaining quantity will have the
    same shape as a, the taken quantity will have the same shape as b.
    '''
    a, b = grids_to_aligned_blocks(a, b)
    
    # min resolution
    existing = jnp.sum(a, axis=(1,3), keepdims=True)
    requested = jnp.sum(b, axis=(1,3), keepdims=True)
    available = jnp.minimum(existing, requested)
    # this use of ratios is to maintain proper shapes
    taken_ratio = jnp.where(requested > eps, available / requested, 1.)
    remaining_ratio = jnp.where(
        existing > eps, (existing - available)/existing, 1.)
    
    # b resolution
    taken = b * taken_ratio
    
    # a resolution
    remaining = a * remaining_ratio
    
    return blocks_to_grid(remaining), blocks_to_grid(taken)

def compare_grids(a, b, mode='<'):
    a, b = grids_to_aligned_blocks(a, b)
    dah, daw = a.shape[1], a.shape[3]
    dbh, dbw = b.shape[1], b.shape[3]
    b = jnp.sum(b, axis=(1,3), keepdims=True)
    b = b/dbh
    b = b/dbw
    
    if mode == '>':
        result = a > b
    elif mode == '>=':
        result = a >= b
    elif mode == '<':
        result = a < b
    elif mode == '<=':
        result = a <= b
    elif mode == '==':
        result = a == b
    
    return blocks_to_grid(result)

def read_grid_locations(a, x, downsample):
    '''
    Read values from specific locations (x) in a downsampled grid (a). The
    downsampled grid values represent the sum of quantities from a higher
    resolution grid, so the values in each coarse cell in a are divided by the
    downsample ratio.
    '''
    xd = x // downsample
    value = a[xd[...,0], xd[...,1]] / (downsample**2)
    return value

def add_to_grid_locations(a, x, value, downsample):
    '''
    Add values to specific locations (x) in a downsampled grid (a).
    '''
    xd = x // downsample
    a = a.at[xd[...,0], xd[...,1]].add(value)
    return a

def take_from_grid_locations(a, x, take, downsample):
    '''
    Subtract values (take) from a downsampled grid (a) at locations (x) while
    enforcing a non-negative constraint on a.  Returns the updated grid a and
    the ammount taken min(a[x], take).  If take is None, the maximum possible
    is taken from each location.
    '''
    taken = read_grid_locations(a, x, downsample)
    taken = jnp.clip(taken, max=take)
    a = add_to_grid_locations(a, x, -taken, downsample)
    return a, taken
