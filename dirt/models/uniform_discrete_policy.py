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
