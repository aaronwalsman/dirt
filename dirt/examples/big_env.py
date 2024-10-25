import time

import jax
import jax.numpy as jnp
import jax.random as jrng
from jax import jit

from flax.struct import dataclass

import chex

from gymnax.environments.environment import Environment

from dirt.dynamics import gridworld2d as dg2d
from dirt.observations import gridworld2d as og2d

@dataclass
class BigEnvParams:
    num_agents : int = 256
    world_size : jax.Array = jax.Array((256,256))
    food_gen : float = 0.05
    max_bite : float = 0.5
    max_food_per_agent : float = 5.
    max_food_per_tile : float = 1.
    observable_area : jax.Array = jax.Array([[0,5],[-2,3]])
    observable_rows_and_cols : jax.Array
    max_steps_in_episode : int = 1
    
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
    time : int
    agent_x : jax.Array
    agent_r : jax.Array
    agent_food : jax.Array
    food_map : jax.Array
    occupancy_map : jax.Array

class BigEnv(Environment):
    
    @property
    def default_params(self) -> BigEnvParams
        return BigEnvParams.create()
    
    def reset_env(
        key: chex.PRNGKey,
        params: Optional
    ) -> Tuple[
        TBigEnvObservation, TBigEnvState, jax.Array, jax.Array, Dict[Any,Any]
    ]:
        
        if params is None:
            
        
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
        
        state = BigEnvState(
            key, agent_x, agent_r, agent_food, food_map, occupancy_map)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: TBigEnvState,
        action: TBigEnvAction,
        params: TBigEnvParams,
    ):
        
        # grow new food
        key, subkey = jrng.split(key)
        food_map = state.food_map + jrng.uniform(
            subkey,
            shape=state.food_map.shape,
            minval=0,
            maxval=system.food_gen,
        )
        
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
        appetite = params.max_food_per_agent - state.agent_food
        array_max_bite = jnp.full_like(appetite, params.max_bite)
        max_eatable_food = jnp.min(
            jnp.stack((appetite, array_max_bite)),
            axis=0,
        )
        available_food = state.food_map[agent_x[:,0], agent_x[:,1]]
        eaten_food = jnp.min(
            jnp.stack((available_food, max_eatable_food)),
            axis=0,
        )
        food_map.at[state.agent_x[:,0], state.agent_x[:,1]].add(-eaten_food)
        agent_food = state.agent_food + eaten_food
        
        # compute the new state
        state = BigEnvState(
            agent_x, agent_r, agent_food, food_map, occupancy_map)
        
        # compute the new observation
        key, obs_key = jrng.split(key)
        obs = self.get_obs(state, params, obs_key)
        
        return obs, state, eaten_food, False, {}

    def get_obs(self, state, params):
        food_view = og2d.extract_fpv(
            state.agent_x,
            state.agent_r,
            params.observable_rows_and_cols,
            state.food_map,
        )
        occupancy_view = og2d.extract_fpv(
            state.agent_x,
            state.agent_r,
            params.observable_rows_and_cols,
            state.occupancy_map,
        ).astype(jnp.float32)
        return jnp.stack((food_view, occupancy_view), axis=1)

if __name__ == '__main__':
    
    key = jrng.key(1234)
    key, action_key = jrng.split(key)
    
    state = reset(key, params)
    
    step = jit(step)
    observe = jit(observe)
    for i in range(10000):
        t1 = time.time()
        
        action_key, subkey = jrng.split(action_key)
        a = jrng.randint(subkey, (system.num_agents, 2), minval=-1, maxval=2)
        state = step(system, state, a)
        o = observe(system, state)
        
        t2 = time.time()
        
        if i % 100 == 0:
            print('fps: ', 1./(t2-t1))
