import jax
import jax.numpy as jnp

from dirt.constants import DEFAULT_FLOAT_DTYPE

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

def downsample_grid_shape(h, w, downsample):
    return h//downsample, w//downsample

def zero_grid(h, w, downsample, cell_shape=(), dtype=DEFAULT_FLOAT_DTYPE):
    dh, dw = downsample_grid_shape(h, w, downsample)
    return jnp.zeros((dh, dw, *cell_shape), dtype=dtype)

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

def downsample_grid(a, h, w, preserve_mass=True):
    '''
    Reduces the size of a grid by summing local blocks of values.
    '''
    a = grid_to_blocks(a, h, w)
    if preserve_mass:
        return jnp.sum(a, axis=(1,3))
    else:
        return jnp.mean(a, axis=(1,3))

def subsample_grid(a, h, w, preserve_mass=True):
    ah, aw = a.shape[:2]
    assert (ah % h == 0) and (aw % w == 0)
    a = grid_to_blocks(a, h, w)
    a = a[:,0,:,0]
    if preserve_mass:
        dh = h // ah
        dw = w // aw
        a = a * (dh * dw)
    return a

def upsample_grid(a, h, w, preserve_mass=True):
    '''
    Increases the size of a grid by repeating and dividing coarse values.
    
    Note: this function is included for the sake of compleness, but should be
    avoided in favor of broadcasting in most cases.
    '''
    ah, aw, *c = a.shape
    assert h >= ah and w >= aw
    assert h % ah == 0 and w % aw == 0
    dh = h // ah
    dw = w // aw
    a = jnp.repeat(jnp.repeat(a, dh, axis=0), dw, axis=1)
    if preserve_mass:
        a = a / (dh*dw)
    
    return a

def set_grid_shape(a, h, w, preserve_mass=True):
    '''
    Upsample or downsample depending on the existing shape
    '''
    if a.shape[0] > h:
        assert a.shape[1] > w
        return downsample_grid(a, h, w, preserve_mass=preserve_mass)
    elif a.shape[0] < h:
        assert a.shape[1] < w
        return upsample_grid(a, h, w, preserve_mass=preserve_mass)
    else:
        return a

def grid_sum_to_mean(a, downsample):
    return a / (downsample**2)

def grid_mean_to_sum(a, downsample):
    return a * (downsample**2)

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
        b = jnp.sum(b, axis=(1,3), keepdims=True) / scale_factor
    else:
        b = jnp.mean(b, axis=(1,3), keepdims=True)
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

def _shift_for_halo(xd, halo):
    if halo == 0:
        return xd
    return xd + halo

def interior_slice(a, halo):
    if halo == 0:
        return a
    return a[halo:-halo, halo:-halo, ...]

def pad_with_halo(a, halo, fill_value=0):
    if halo == 0:
        return a
    return jnp.pad(
        a,
        ((halo, halo), (halo, halo), *(((0, 0),) * (a.ndim - 2))),
        mode='constant',
        constant_values=fill_value,
    )

def set_interior(a, interior, halo):
    if halo == 0:
        return interior
    return a.at[halo:-halo, halo:-halo, ...].set(interior)

def make_grid(
    world_size,
    downsample=1,
    halo=0,
    cell_shape=(),
    dtype=DEFAULT_FLOAT_DTYPE,
    init_value=0,
    fill_value=0,
):
    assert halo >= 0
    dh, dw = downsample_grid_shape(*world_size, downsample)
    full_shape = (dh + 2 * halo, dw + 2 * halo, *cell_shape)
    interior_shape = (dh, dw, *cell_shape)

    def _fill(grid, value):
        if halo == 0:
            return grid.at[:,:].set(value)
        return grid.at[halo:-halo, halo:-halo, ...].set(value)

    class Grid:
        def init():
            grid = jnp.full(full_shape, fill_value, dtype=dtype)
            return _fill(grid, init_value)

        def interior(grid):
            return interior_slice(grid, halo)

        def set_interior(grid, new):
            return set_interior(grid, new, halo)

        def read(grid, x, downsample_scale=True):
            return read_grid_locations(
                grid, x, downsample, downsample_scale=downsample_scale, halo=halo)

        def write(grid, x, value, overwrite_all=False):
            return write_grid_locations(
                grid, x, value, downsample, overwrite_all=overwrite_all, halo=halo)

        def add(grid, x, value):
            return add_to_grid_locations(grid, x, value, downsample, halo=halo)

        def take(grid, x, take=None):
            return take_from_grid_locations(grid, x, take, downsample, halo=halo)

        def sum_to_mean(grid):
            return grid_sum_to_mean(grid, downsample)

        def mean_to_sum(grid):
            return grid_mean_to_sum(grid, downsample)

        def set_shape(grid, h, w, preserve_mass=True):
            return set_grid_shape(grid, h, w, preserve_mass=preserve_mass)

        def downsample_to(grid, h, w, preserve_mass=True):
            return downsample_grid(grid, h, w, preserve_mass=preserve_mass)

        def upsample_to(grid, h, w, preserve_mass=True):
            return upsample_grid(grid, h, w, preserve_mass=preserve_mass)

    Grid.init = staticmethod(Grid.init)
    Grid.interior = staticmethod(Grid.interior)
    Grid.set_interior = staticmethod(Grid.set_interior)
    Grid.read = staticmethod(Grid.read)
    Grid.write = staticmethod(Grid.write)
    Grid.add = staticmethod(Grid.add)
    Grid.take = staticmethod(Grid.take)
    Grid.sum_to_mean = staticmethod(Grid.sum_to_mean)
    Grid.mean_to_sum = staticmethod(Grid.mean_to_sum)
    Grid.set_shape = staticmethod(Grid.set_shape)
    Grid.downsample_to = staticmethod(Grid.downsample_to)
    Grid.upsample_to = staticmethod(Grid.upsample_to)

    Grid.world_size = world_size
    Grid.downsample = downsample
    Grid.halo = halo
    Grid.cell_shape = cell_shape
    Grid.dtype = dtype
    Grid.init_value = init_value
    Grid.fill_value = fill_value
    Grid.full_shape = full_shape
    Grid.interior_shape = interior_shape

    return Grid

def read_grid_locations(a, x, downsample, downsample_scale=True, halo=0):
    '''
    Read values from specific locations (x) in a downsampled grid (a). The
    downsampled grid values represent the sum of quantities from a higher
    resolution grid, so the values in each coarse cell in a are divided by the
    downsample ratio.
    '''
    xd = x // downsample
    xd = _shift_for_halo(xd, halo)
    value = a.at[xd[...,0], xd[...,1]].get(mode='fill', fill_value=0.)
    if downsample_scale:
        value /= (downsample**2)
    return value

def write_grid_locations(
    a, x, value, downsample, overwrite_all=False, halo=0):
    '''
    Write values to specific locations (x) in a downsampled grid (a).  The
    downsampled grid values represent the sum of quantities from a higher
    resolution grid.  If overwrite_all is set to True, this will overwrite
    all value in each coarse grid cell with value * (downsample**2).
    Otherwise a portion of the existing value in each grid cell will be
    removed before the the new value is added in.  This means that a grid
    cell which as been downsampled by 2, and had it's value written by this
    function will have 3/4 of its original value plus the value specified for
    this cell after this operation. 
    '''
    xd = x // downsample
    xd = _shift_for_halo(xd, halo)
    xd0 = xd[...,0]
    xd1 = xd[...,1]
    if overwrite_all:
        a = a.at[xd0, xd1].set(value * downsample**2)
    else:
        current_value = a[xd0, xd1]
        a = a.at[xd0, xd1].add(
            value - current_value * (1./(downsample**2)))
    
    return a

def add_to_grid_locations(a, x, value, downsample, halo=0):
    '''
    Add values to specific locations (x) in a downsampled grid (a).
    '''
    xd = x // downsample
    xd = _shift_for_halo(xd, halo)
    a = a.at[xd[...,0], xd[...,1]].add(value)
    return a

def take_from_grid_locations(a, x, take, downsample, halo=0):
    '''
    Subtract values (take) from a downsampled grid (a) at locations (x) while
    enforcing a non-negative constraint on a.  Returns the updated grid a and
    the ammount taken min(a[x], take).  If take is None, the maximum possible
    is taken from each location.
    '''
    taken = read_grid_locations(a, x, downsample, halo=halo)
    if take is not None:
        taken = jnp.clip(taken, max=take)
    a = add_to_grid_locations(a, x, -taken, downsample, halo=halo)
    return a, taken
