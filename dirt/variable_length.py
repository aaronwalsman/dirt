import jax
import jax.numpy as jnp

def running_ones(n, m, start=0):
    return jnp.arange(m) < (n+start)

def compact(i, x, empty_value=-1):
    empty = (i == empty_value)
    nonempty = ~empty
    
    # compute the destination indices for each nonempty value
    nonempty_perm = jnp.cumsum(nonempty)-1
    
    # compute the destination indices for each empty value
    n = nonempty_perm[-1]
    empty_perm = jnp.cumsum(empty) + n
    
    # combine the empty and nonempty indices
    perm = nonempty_perm * nonempty + empty_perm * empty
    
    # permute the original values
    ci = i.at[perm].set(i)
    cx = jax.tree.map(lambda l: l.at[perm].set(l), x)
    
    return ci, cx

def concatenate(i, x, max_length=None, empty_value=-1):
    if max_length is None:
        max_length = i[0].shape[0]
    i = jnp.concatenate(i)
    f = lambda *x : jnp.concatenate(x)
    x = jax.tree.map(f, *x)
    ci, cx = compact(i, x, empty_value=empty_value)
    ci = ci[:max_length]
    cx = jax.tree.map(lambda l: l[:max_length], cx)
    
    return ci, cx

def compact_scan(i, x, max_n, empty_value=-1):
    def step(compact_state, ixj):
        # unpack
        ci, cx, write_index = compact_state
        ij, xj = ixj
        
        # write the 
        ci = ci.at[write_index].set(ij)
        cx = tree.map(
            lambda cleaf, leaf : cleaf.at[write_index].set(leaf),
            (cx, xj),
        )
        
        # increment the write index if necessary
        write_index += (xi != empty_index)
        
        return (ci, cx, write_index), None
    
    cids = jnp.full(max_n, empty_value)
    cx = jax.tree.map(lambda leaf : jnp.zeros_like(leaf))
    (cids, cx, _), _ = jax.lax.scan(step, (cids, cx, 0), (i, x))
    return cx

'''
def compact_concatenate(arrays, max_n=None, empty_value=-1):
    if max_n is None:
        max_n = arrays[0].shape[0]
    x = jnp.concatenate(arrays, axis=0)
    return compact(x, max_n, empty_value=empty_value)
'''
