import jax.numpy as jnp

'''
We have an "alive" array, and an array that keeps track of each agent's
birthday.  Or maybe just the birthday with -1 indicating no agent.
The new idea is that instead of having increasing dynamic logical indices,
instead if players never change their slot in the player array, we can index
them using their physical index plus their birthday.  This means that we will
have lots of gaps, but that's totally fine if it frees us from having to
compact stuff or compute global information over the entire list.

Ok, so we just track birthdays, since the physical index is already implicit.
Then what about parent information?  This is just a 2D array containing parent
physical index plus birthday.

Yeah, easy.

So do we even need this file?  We're just doing a scatter right?

The operation is:

current_time = 3
birthdays =          [-1, 0,-1, 0, 1, 2,-1,-1]
parent_birthdays =   [-1,-1,-1,-1, 0, 0,-1,-1]
parent_indices =     [-1,-1,-1,-1, 0, 1,-1,-1]
reproduce =          [ 0, 1, 0, 1, 0, 1, 0, 0]

scatter_indices = jnp.nonzero(birthdays == -1, size=8, fill_value=8)
scatter_indices = scatter_indices * leading_ones(jnp.sum(reproduce))
birthdays.at[scatter_indices].set(current_time)
new_parent_birthdays = birthdays.at[scatter_indices].get(
    mode='fill', fill_value=-1)
parent_birthdays.at[scatter_indices].set(new_parent_birthdays)
reproduce_indices = jnp.nonzero(reproduce, size=8, fill_value=8)
parent_indices.at[scatter_indices].set(reproduce_indices)

So that is a little bit nontrivial.  It would be nice to do this with supplied
data as well.  It would also be nice if the player/parent info returned by
the population game made it easy to copy stuff around without needing to redo
this computation.  Like is there any way we could just return what's new and
where it came from?  Then if you wanted to trace geneology, you could do it
by hand?  With that in mind, what you'd be doing is returning a single array
instead of players and parents.  The array would be -1 everywhere except where
a new baby shows up, and the parent value for each of those new baby locations.

Wait... does that make the computation above easier too.  Or is that what I am
already computing above?  Yeah it's basically this part:

scatter_indices = jnp.nonzero(birthdays == -1, size=8, fill_value=8)
scatter_indices = scatter_indices * leading_ones(jnp.sum(reproduce))
reproduce_indices = jnp.nonzero(reproduce, size=8, fill_value=8)
new_parent_indices = jnp.full_like(scatter_indices, -1)
new_parent_indices.at[scatter_indices].set(reproduce_indices) # return this

Oh, so maybe what we're saying is don't even bother with birthdays?  We don't
need them in the system.  If you want to track geneology, go nuts man, but
we don't have to do that in the simulator, you can do that in the experimental
code using the sequence of parent_indices.  We honestly don't even need to
track parents long term, we can just return it once.  So we don't keep track
of long-term identity in the simulator at all, just which physical indices are
alive.

Ok, so now, if you are given a list of new_parents, how do you use it?
write_locations = jnp.nonzero(new_parent_indices != 8, size=..., fill_value=...)
my_big_arrays.at[write_locations].set(my_big_arrays.at[BLAH])

Wait, what do I actually want it to return?  Two arrays: a set of parents to
read from, and a set of locations to write to.  Maybe the parents could
generalize to Nxk, if it's possible to have k parents.  Whatever.  And then
maybe also a third array for who's alive?  But wait, again, it seems like those
read and write locations should be specifyable using a single array... right?
The question is, can it be a single array with minimal computation?  And the
answer is probably not, you'd have to run nonzero again.  So two arrays?
I still feel like there should be some indexing hack (or non-hack) to do this
easily.  Oh wait, is it just scatter?  No wait, scatter is just .at[].set().

=========================================================

Ok, so we don't track birthdays, we just keep track of who's alive and who
isn't, and then return two arrays: who's currently alive, and the parents of
new babies.  This would look like:

previous_alive = [ 0, 1, 1, 0, 0, 1, 0, 0]

new_alive =      [ 0, 0, 1, 1, 1, 1, 0, 0]
new_parents =    [-1,-1,-1, 1, 5,-1,-1,-1]

From this you can track deaths by comparing successive alive vectors.
You can copy data using:

previous_data =  [ 0, 1, 2, 0, 0, 5, 0, 0]
padded_np =      jax.where(new_parents == -1, 8, new_parents)
new_data =       previous_data.at[new_parents != -1].set(
                    previous_data.at[padded_np].get(fill whatever))

For future purposes (brifely) what would it be if you wanted to average over
your parents?

previous_alive = [ 0, 1, 1, 0, 0, 1, 1, 0]
new_parents =    [-1,-1,-1, 1, 5,-1,-1,-1]
                 [-1,-1,-1, 2, 6,-1,-1,-1]

So then you need to make a 2x8xc tensor where you use the same trick to put
everything into the right place, then simply sum and average?
'''

def inherit_single_parent_tensor(
    data,
    parents,
    empty=-1,
    mutate=None,
):
    empty_parents = (parents == empty)
    parent_data = data[parents]
    if mutate is not None:
        parent_data = mutate(parent_data)
    inherited_data = jnp.where(empty_parents, data, parent_data)
    
    return inherited_data

def inherit_multi_parent_tensor(
    data,
    parents,
    empty=-1,
    mutate=None,
    crossover=None,
):
    v_inherit = vmap(inherit_single_parent_tensor, in_axes=(None,1,None,None))
    inherited_data = v_inherit(data, parents, empty=empty, mutate=mutate)
    if crossover is not None:
        v_crossover = vmap(crossover, in_axes=(0,))
        inherited_data = v_crossover(inherited_data)
    
    return inherited_data

def inherit_single_parent_tree(
    data,
    parents,
    empty=-1,
    mutate=None,
):
    def inherit(leaf):
        return inherit_single_parent_tensor(
            leaf, parents, empty=empty, mutate=mutate)
    
    return jax.tree.map(inherit, data)

def inherit_multi_parent_tree(
    data,
    parents,
    empty=-1,
    mutate=None,
    crossover=None,
):
    def inherit(leaf):
        return inherit_multi_parent_tensor(
            leaf, parents, empty=empty, mutate=mutate, crossover=crossover)
    
    return jax.tree.map(inherit, data)

def align_parent_data(data, parents, empty=-1):
    WRONG_USE_VMAP
    if len(parents.shape) == 1:
        parents = parents[:,None]
        remove_axis = True
    else:
        remove_axis = False
    
    n, k = parents.shape
    nonempty = jnp.nonzero(parents!=empty, size=n, fill_value=n*k)
    parent_data = data[:,None].at[nonempty].set(data)
    breakpoint()
    return parent_data

#def inherit_from_single_parent(data, parents, empty=-1, operation=lambda x : x):
#    n, = parents.shape
#    empty_parents = (parents == empty)
#    
#    def operate(d):
#        z = jnp.zeros_like(d)
#        z.
#    scatter = jax.tree.map(
#        operate,
#        data,
#    )
#    data = jax.tree.map(
#        lambda s : jax.where(parents==empty, parents, s),
#        scatter,
#    )
#        
#    return data

def inherit_from_multiple_parents(data, parents, empty=-1, operation=jnp.mean):
    n,k = parents.shape
    parents = jax.where(new_parents==empty, n, new_parents)
    #data = jax.where(parents!=empty, 

def merge(i0, i1, x0, x1, empty=-1):
    n = i.shape[0]
    empty_indices = jnp.nonzero(i0==empty, size=n, fill_value=n)

def reproduce(ids, reproduce, write, empty=-1):
    n = ids.shape[0]
    empty_slots, = jnp.nonzero(ids == empty, size=n, fill_value=n)
    empty_slots = jnp.where(jnp.arange(n) < jnp.sum(reproduce), empty_slots, n)
    
    r, = jnp.nonzero(reproduce, size=n, fill_value=n)
    w = write.at[r].get(mode='fill', fill_value=n)
    
    ids = ids.at[empty_slots].set(write)
    breakpoint()

if __name__ == '__main__':
    #ids = jnp.array([-1,1,2,-1,-1,-1,1,3])
    #r = jnp.array([0, 1, 0, 0, 0, 0, 0, 1])
    #write = jnp.array([0,4,0,0,0,0,0,5])
    #reproduce(ids, r, write)
    
    '''
    data = jnp.arange(8)
    parents = jnp.array([
        [-1,0,1,1,-1,-1,-1,-1],
        [-1,1,2,2,-1,-1,-1,-1],
    ]).T
    new_data = align_parent_data(data, parents)
    '''
    
    data = jnp.arange(8)
    parents = jnp.array([-1,0,1,0,-1,-1,-1,-1])
    new_data = inherit_single_parent_tensor(data, parents)
    breakpoint()
