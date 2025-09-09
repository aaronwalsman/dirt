import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static import static_functions

def make_uniform_discrete_policy(num_actions):
    @static_functions
    class UniformDiscretePolicy:
        def init():
            return jnp.array(num_actions, dtype=jnp.int32)
        
        def act(key, state):
            action = jrng.randint(key, (), minval=0, maxval=state)
            return action
    
    return UniformDiscretePolicy

def make_random_discrete_policy(p_actions):
    @static_functions
    class RandomDiscretePolicy:
        def init():
            return p_actions
        
        def act(key, state):
            action = jrng.choice(key, p_actions.shape[0], (), p=p_actions)
            return action
    
    return RandomDiscretePolicy
