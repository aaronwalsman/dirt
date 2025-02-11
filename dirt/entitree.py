'''
What would this be?

I basically want a jax dictionary, where I can use arbitrary logical indices
to refer to entries in a tree of data.  The only operations I want is to be
able to look up and set tree data based on the logical indices.  So like
setitem, getitem functionality, although it doesn't need to be wrapped into
a class like that.  Oh, I guess we also want delete (del) and length (len)
functionality.  The last thing we need is to also be able to get and set and
delete using some kind of masked arrays so that we can use fixed-sized data
everywhere.

One quick thing: we can also break setitem into two parts.  One operation
inserts indices that you can guarantee don't already exist.  Another operation
sets indices that you can guarantee do exist.

Lookup is easy.  Let's assume that logical indices are integers (but could
easily be extended to NxK integer arrays, where each location has a tuple
of K indices).  Let's say we have all the extant indices in an N-length
array, where a logical index's physical location is indicated by it's location
in the array (this array is a physical-to-logical lookup).  We just use
a jnp.nonzero(indices == lookup) call to get the physical indices, and then
use those to lookup.  The lookup should also store invalid locations.

Deleting is also not too bad, we just set the corresponding physical locations
to be invalid.

Setting is where things get challenging.  We need to find a list of free
logical indices.  This can be done with jnp.nonzero(invalid) with the
appropriate fill values.
'''



def insert_physical_locations(mask, insert_mask, max_new):
    n, = mask.shape
    m, = insert_mask.shape
    available_locations = jnp.nonzero(~mask, size=max_new, fill_value=n)
    insert_indices = jnp.nonzero(insert_mask, size=max_new, fill_value=n)
    z = jnp.zeros(n, dtype=jnp.int32)
    z.at[insert_indices] = available_locations
    return z
