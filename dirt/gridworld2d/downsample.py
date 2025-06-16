'''
def align_dense_to_blocks(a, b) #, scale_b=True):
    ha, wa, *ca = a.shape
    hb, wb, *cb = b.shape
    
    if ha == hb:
        dha = 1
        dhb = 1
    elif ha > hb:
        assert ha % hb == 0
        dha = ha // hb
        dhb = 1
    else:
        assert hb % ha == 0
        dha = 1
        dhb = hb // ha
    
    if wa == wb:
        dwa = 1
        dwb = 1
    elif wa > wb:
        assert wa % wb == 0
        dwa = wa // wb
        dwb = 1
    else:
        assert wb % wa == 0
        dwa = 1
        dwb = wb // wa
    
    a = dense_to_blocks(a, dha, dwa)
    b = dense_to_blocks(b, dhb, dwb)
    
    return a, b
'''

def align_downsampled_grid(x, dh, dw):
    h, w, *c = x.shape
    assert h % dh == 0 and w % dw == 0
    return x.reshape(h//dh, dh, w//dw, dw, *c)

def unalign(a):
    h, dh, w, dw, *c = a.shape
    return a.reshape(h*dh, w*dw, *c)

def align_downsampled_grids(*downsampled_grids):
    hs, ws, cs = zip(
        [[d.shape[0], d.shape[1], d.shape[2:]] for d in downsampled_grids])
    min_h = min(hs)
    min_w = min(ws)
    return tuple(
        align_downsampled_grid(d, min_h, min_w) for d in downsampled_grids)

#def add_dense(a, b):
#    a, b = align_dense_to_blocks(a, b)
#    b = jnp.sum(b, axis=(1,3), keepdim=True)
#    ab = a + b
#    return blocks_to_dense(ab)

'''
(32,32) -> (16,2,16,2)
(64,64) -> (16,4,16,4) # I want to only sum this one over a 2, not 4 values
(16,16) -> (16,1,16,1) # I also want to divide the values here by 1/4

(32,32) -> (32,1,32,1)
(64,64) -> (32,2,32,2)
(16,16) -> (32,1,32,1) # would need to repeat this and again divide by 1/4
'''

def mask_aligned(a, m, mode='any'):
    if ha == hb:
        assert wa == wb
        return a * s
    elif ha > hm:
        # assert
        # reshape a
        # c = a * s
        # reshape c
    elif hm > ha:
        # assert
        # reshape s
        # apply any/all
        # c = a * s
    return c

'''
2 1   4 4
3 2 * 4 4 = 8+4+12+8 = 32 / 4 = 8
'''

def scale_downsampled_grids(x, alpha):
    x, alpha = align_downsampled_grids(x, alpha)
    dxh, dxw = x.shape[1], x.shape[3]
    scale_factor = 1. / alpha.shape[1] * alpha.shape[3]
    x = x * scale_factor * alpha
    if dxh == 1 and dxw == 1:
        x = jnp.sum(x, axis=(1,3))
    x = dealign(x)
    return x

def add_downsampled_grids(a, b):
    ha, wa, *ca = a.shape
    hb, wb, *cb = b.shape
    if ha == hb:
        assert wa == wb
        return a + b
    elif ha > hb:
        # assert
        # reshape a
        # c = a + b * scale**2
        # reshape c
    elif hb > ha:
        # assert
        # reshape b
        # sum b
        # c = a + b
    
    return c

def take_aligned(a, b):
    

def add_aligned(a, *b):
    a, *b = align_to_minimum(a, *b)
    b = (jnp.sum(bi, axis=(1,3), keepdim=True) for bi in b)
    c = sum(a, *b)

def read_from_block(grid, x, downsample):
    xd = x // downsample
    value = grid[xd[...,0], xd[...,1]] / (downsample**2)
    return value

def add_to_block(grid, x, value, downsample):
    xd = x // downsample
    grid = grid.at[xd[...,0], xd[...,1]].add(value)
    return grid

def take_from_block(grid, x, downsample, max=None):
    value = get_from_block_grid(grid, x, downsample)
    value = jnp.clip(value, max=max)
    grid = add_to_block_grid(grid, x, -value, downsample)
    return grid, value
