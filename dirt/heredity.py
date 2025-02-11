import jax
import jax.numpy as jnp

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
    v_inherit = jax.vmap(
        inherit_single_parent_tensor, in_axes=(None,1, None, None))
    inherited_data = v_inherit(data, parents, empty, mutate)
    if crossover is not None:
        v_crossover = jax.vmap(crossover, in_axes=(1,))
        inherited_data = v_crossover(inherited_data)
    
    return inherited_data

#def inherit_single_parent_tree(
#    data,
#    parents,
#    empty=-1,
#    mutate=None,
#):
#    def inherit(leaf):
#        return inherit_single_parent_tensor(
#            leaf, parents, empty=empty, mutate=mutate)
#    
#    return jax.tree.map(inherit, data)
#
#def inherit_multi_parent_tree(
#    data,
#    parents,
#    empty=-1,
#    mutate=None,
#    crossover=None,
#):
#    def inherit(leaf):
#        return inherit_multi_parent_tensor(
#            leaf, parents, empty=empty, mutate=mutate, crossover=crossover)
#    
#    return jax.tree.map(inherit, data)

def test_inherit_single_parent_tensor():
    data = jnp.arange(8)
    parents = jnp.array([-1,0,1,0,-1,-1,-1,-1])
    mutate = lambda x : x + 1
    inherited_data = inherit_single_parent_tensor(data, parents, mutate=mutate)
    
    assert jnp.all(inherited_data == jnp.array([0,1,2,1,4,5,6,7]))

def test_inherit_multi_parent_tensor():
    data = jnp.arange(8).astype(jnp.float32)
    parents = jnp.array([
        [-1,-1],
        [ 0, 4],
        [ 0, 5],
        [ 4, 7],
        [-1,-1],
        [-1,-1],
        [-1,-1],
        [-1,-1],
    ])
    mutate = lambda x : x + 1
    crossover = jnp.mean
    inherited_data = inherit_multi_parent_tensor(
        data, parents, mutate=mutate, crossover=crossover)
    
    assert jnp.all(inherited_data == jnp.array(
        [0. , 3. , 3.5, 6.5, 4. , 5. , 6. , 7. ], dtype=jnp.float32))

def test_inherit_single_parent_tree():
    data = (jnp.arange(8), jnp.arange(8,16))
    parents = jnp.array([-1,0,1,0,-1,-1,-1,-1], dtype=jnp.int32)
    mutate = lambda x : x + 1
    inherited_data = inherit_single_parent_tree(data, parents, mutate=mutate)
    
    assert jnp.all(inherited_data[0] == jnp.array([ 0, 1, 2, 1, 4, 5, 6, 7]))
    assert jnp.all(inherited_data[1] == jnp.array([ 8, 9,10, 9,12,13,14,15]))

def test_inherit_multi_parent_tree():
    data = (jnp.arange(8), jnp.arange(8,16))
    parents = jnp.array([
        [-1,-1],
        [ 0, 4],
        [ 0, 5],
        [ 4, 7],
        [-1,-1],
        [-1,-1],
        [-1,-1],
        [-1,-1],
    ])
    mutate = lambda x : x + 1
    crossover = jnp.mean
    inherited_data = inherit_multi_parent_tree(
        data, parents, mutate=mutate, crossover=crossover)
    
    assert jnp.all(inherited_data[0] ==
        jnp.array([ 0. , 3. , 3.5, 6.5, 4. , 5. , 6. , 7. ]))
    assert jnp.all(inherited_data[1] ==
        jnp.array([ 8. , 11. , 11.5, 14.5, 12. , 13. , 14. , 15. ]))

def available_child_ids_old(
    population_alive,
    child_mask,
    max_children=None,
):
    n, = population_alive.shape
    m, = child_mask.shape
    available_locations = jnp.nonzero(
        ~population_alive, size=max_children, fill_value=n)
    child_mask_locations = jnp.nonzero(
        child_mask, size=max_children, fill_value=n)
    child_ids = jnp.zeros(n, dtype=jnp.int32)
    child_ids.at[child_mask_locations] = available_locations
    return child_ids

def available_child_ids(
    population_alive,
    num_children,
):
    n, = population_alive.shape
    available_locations = jnp.nonzero(
        ~population_alive, size=num_children, fill_value=n)
    return available_locations

def produce_children(
    parents,
    player_data,
    birth_process,
    empty=-1,
):
    # constants
    max_children, parents_per_child = parents.shape
    
    # get the valid children
    valid_children = jnp.all(parents != empty, axis=1)
    
    # get the parent data
    def get_multi_parent_data(leaf):
        _, *leaf_shape = leaf.shape
        return leaf[parents.reshape(-1)].reshape(
            max_children, parents_per_child, *leaf_shape)
    parent_data = jax.tree.map(get_multi_parent_data, player_data) 
    
    # apply the birth process
    child_data = birth_process(parent_data)
    
    return child_data
    
'''
    # assign the child data to the appropriate locations
    child_ids = available_child_ids(
        alive, valid_children, max_children)
    def set_child_data(player_leaf, child_leaf):
        return player_leaf.at[child_ids].set(child_leaf)
    player_data = jax.tree.map(set_child_data, player_data, child_data)
    
    return player_data, child_ids
'''

if __name__ == '__main__':
    
    test_inherit_single_parent_tensor()
    test_inherit_multi_parent_tensor()
    test_inherit_single_parent_tree()
    test_inherit_multi_parent_tree()
