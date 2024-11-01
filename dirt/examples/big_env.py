import time

import jax
import jax.numpy as jnp
import jax.random as jrng
from jax import jit

from flax.struct import dataclass

from dirt.dynamics import gridworld2d as dg2d
from dirt.observations import gridworld2d as og2d

@dataclass
class BigEnvSystem:
    num_agents : jax.Array
    world_size : jax.Array
    food_gen : jax.Array
    max_bite : jax.Array
    max_food_per_agent : jax.Array
    max_food_per_tile : jax.Array
    observable_area : jax.Array
    observable_rows_and_cols : jax.Array
    
    @classmethod
    def create(cls, *args, observable_area, **kwargs):
        a, b = observable_area.T
        s = jnp.sign(b-a)
        observable_rows_and_cols = jnp.stack(jnp.meshgrid(
            jnp.arange(a[0], b[0], s[0]),
            jnp.arange(a[1], b[1], s[1]),
            indexing='ij',
        ), axis=-1)
        return cls(
            *args,
            observable_area=observable_area,
            observable_rows_and_cols=observable_rows_and_cols,
            **kwargs,
        )

@dataclass
class BigEnvState:
    key : jax.Array
    agent_x : jax.Array
    agent_r : jax.Array
    agent_food : jax.Array
    food_map : jax.Array
    occupancy_map : jax.Array

def reset(key, system):
    
    # generate initial agent positions
    key, subkey = jrng.split(key)
    agent_x = jrng.randint(
        subkey,
        shape=(system.num_agents, 2),
        minval=jnp.array([0,0]),
        maxval=system.world_size,
    )
    key, subkey = jrng.split(key)
    agent_r = jrng.randint(
        subkey,
        shape=(system.num_agents,),
        minval=0,
        maxval=4,
    )
    
    # generate initial agent food
    agent_food = jnp.ones(system.num_agents)
    
    # generate initial food map
    key, subkey = jrng.split(key)
    food_map = jrng.uniform(
        subkey,
        shape=system.world_size,
        minval=0.,
        maxval=system.max_food_per_tile,
    )
    
    # generate occupancy map
    occupancy_map = jnp.zeros(system.world_size, dtype=jnp.int32)
    
    return BigEnvState(
        key, agent_x, agent_r, agent_food, food_map, occupancy_map)

def step(system, state, action):
    
    # grow food
    key, subkey = jrng.split(state.key)
    food_map = state.food_map + jrng.uniform(
        subkey, shape=state.food_map.shape, minval=0, maxval=system.food_gen)
    
    # move the agents
    dx = jnp.array([1,0])[None,...] * action[...,0,None]
    dr = action[...,1]
    (
        agent_x,
        agent_r,
        _,
        occupancy_map,
    ) = dg2d.step_clip_collide(
        state.agent_x,
        dx,
        state.agent_r,
        dr,
        state.occupancy_map,
    )
    
    # eat the food
    appetite = system.max_food_per_agent - state.agent_food
    array_max_bite = jnp.full_like(appetite, system.max_bite)
    max_eatable_food = jnp.min(
        jnp.stack((appetite, array_max_bite)),
        axis=0,
    )
    available_food = state.food_map[agent_x[:,0], agent_x[:,1]]
    eaten_food = jnp.min(
        jnp.stack((available_food, max_eatable_food)),
        axis=0,
    )
    food_map.at[state.agent_x[:,0], state.agent_x[:,1]].add(
        -eaten_food)
    agent_food = state.agent_food + eaten_food
    
    return BigEnvState(
        key, agent_x, agent_r, agent_food, food_map, occupancy_map)

def observe(system, state):
    food_view = og2d.extract_fpv(
        state.agent_x,
        state.agent_r,
        system.observable_rows_and_cols,
        state.food_map,
    )
    occupancy_view = og2d.extract_fpv(
        state.agent_x,
        state.agent_r,
        system.observable_rows_and_cols,
        state.occupancy_map,
    ).astype(jnp.float32)
    return jnp.stack((food_view, occupancy_view), axis=1)

if __name__ == '__main__':
    k = 1024*8
    system = BigEnvSystem.create(
        num_agents = jnp.array(k),
        world_size = jnp.array([k,k]),
        food_gen = jnp.array(0.05),
        max_bite = jnp.array(0.5),
        max_food_per_agent = jnp.array(5.),
        max_food_per_tile = jnp.array(1.),
        observable_area = jnp.array([[0,5],[-2,3]])
    )
    
    key = jrng.key(1234)
    key, action_key = jrng.split(key)
    
    state = reset(key, system)
    
    step = jit(step)
    observe = jit(observe)
    i = 0
    while True:
        i += 1
        t1 = time.time()
        
        action_key, subkey = jrng.split(action_key)
        a = jrng.randint(subkey, (system.num_agents, 2), minval=-1, maxval=2)
        state = step(system, state, a)
        o = observe(system, state)
        
        t2 = time.time()
        
        if i % 100 == 0:
            print('fps: ', 1./(t2-t1))
