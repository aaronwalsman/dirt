'''
Goal: Figure out how to distribute this across multiple GPUs effectively.
What we want is to support very large world_size and num_agents.

There are a few different pieces of data and computation that needs to be
distributed:
1. The environment (DistributedEnvState) contains data that scales with
    world_size (food_grid, object_grid), but also contains data that scales
    with num_agents (agent_x, agent_r, agent_food).
2. The model (population_state) contains larger data (network weights) that
    scales with num_agents.

Computationally, data is passed back and forth between world_size shaped data
and num_agents shaped data at various points.  In the environmental dynamics,
resources (food) are exchanged between the agents and the map.  Observations
are also extracted from the map and passed to each agent.

I have tried to make this as self-contained as possible, but did end up using
a few different existing components:
1. mechagogue nn for network components
2. dirt.gridworld2d.spawn/dynamics/observations
These may need to change based on how we distribute data/computation, but we
should try to keep the functionality ~intact.
'''

# typing
from typing import Tuple

# jax
import jax
import jax.numpy as jnp
import jax.random as jrng

# commandline and static data utilities
from mechagogue.commandline import commandline_interface
from mechagogue.static_dataclass import static_dataclass

# mechagogue nn
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.linear import linear_layer
from mechagogue.nn.distributions import categorical_sampler_layer

# some basic gridworld dynamics
import dirt.gridworld2d.spawn as spawn
import dirt.gridworld2d.dynamics as dynamics
import dirt.gridworld2d.observations as observations

# this can be changed if necessary
FLOAT_DTYPE = jnp.bfloat16

@static_dataclass
class DistributedEnvParams:
    world_size : Tuple[int,int] = (256,256)
    total_food : int = 64**2
    num_agents : int = 256

@static_dataclass
class DistributedEnvState:
    food_grid : jnp.ndarray
    agent_x : jnp.ndarray
    agent_r : jnp.ndarray
    agent_food : jnp.ndarray
    object_grid : jnp.ndarray

def distributed_env(params):
    
    def reset(key):
        # generate the food grid, this will be 2D tensor of shape (world_size)
        # that represents where food exists in the environment
        food_grid = jnp.zeros(params.world_size, dtype=FLOAT_DTYPE)
        key, food_key = jrng.split(key)
        food_x = spawn.unique_x(food_key, params.total_food, params.world_size)
        food_grid = food_grid.at[food_x[...,0], food_x[...,1]].set(1.)
        
        # generate the agent positions (agent_x) and rotations (agent_r) and
        # food (agent_food) that represent each agent's location in the world
        # and how much food they currently have.
        key, agent_x_key, agent_r_key = jrng.split(key, 3)
        agent_x = spawn.unique_x(
            agent_x_key, params.num_agents, params.world_size)
        agent_r = spawn.uniform_r(agent_r_key, params.num_agents)
        agent_food = jnp.zeros((params.num_agents,), dtype=FLOAT_DTYPE)
        
        # generate the object grid, this will be a 2D tensor of shape
        # (world_size) that also keeps track of where the agents are in the
        # map and is used to prevent collisions later
        object_grid = jnp.full(params.world_size, -1, dtype=jnp.int32)
        object_grid = object_grid.at[agent_x[...,0], agent_x[...,1]].set(
            jnp.arange(params.num_agents))
        
        # bundle the data into a state object
        state = DistributedEnvState(
            food_grid,
            agent_x,
            agent_r,
            agent_food,
            object_grid,
        )
        
        # extract the first-person view for each player
        observation = observations.first_person_view(
            agent_x,
            agent_r,
            food_grid,
            view_width=5,
            view_distance=5,
        )
        
        return state, observation
    
    def step(key, state, action):
        
        # deposit some of the agents' existing food to the square that they
        # are leaving (return resources to the environment)
        # - figure out how much will be deposited (min(0.1, agent_food))
        deposited_food = jnp.where(
            state.agent_food < 0.1, state.agent_food, 0.1)
        # - subtract from the agent's internal food
        agent_food = state.agent_food - deposited_food
        # - add deposited food back to the grid
        food_grid = state.food_grid.at[
            state.agent_x[...,0], state.agent_x[...,1]].add(deposited_food)
        
        # move the agents based on the action (agent_x, agent_r and object_grid
        # will now represent the new locations of the agents)
        forward = (action == 1).astype(jnp.int32)
        rotate = (
            (action == 2).astype(jnp.int32) - (action == 3).astype(jnp.int32)
        )
        agent_x, agent_r, c, object_grid = dynamics.forward_rotate_step(
            state.agent_x,
            state.agent_r,
            forward,
            rotate,
            check_collisions=True,
            object_grid=state.object_grid,
        )
        
        # make the agents eat whatever food is in their current location
        # (pull resources from the environment into the agent)
        eaten_food = food_grid[agent_x[...,0], agent_x[...,1]]
        # - subtract the food from the environment
        food_grid = food_grid.at[agent_x[...,0], agent_x[...,1]].set(0.)
        # - add the food to the agent
        agent_food = agent_food + eaten_food
        
        # bundle the new state variables
        next_state = DistributedEnvState(
            food_grid,
            agent_x,
            agent_r,
            agent_food,
            object_grid,
        )
        
        # compute the new observation by extracting local regions of the map
        observation = observations.first_person_view(
            agent_x,
            agent_r,
            food_grid,
            view_width=5,
            view_distance=5,
        )
        
        return next_state, observation
    
    return reset, step

@commandline_interface
@static_dataclass
class TrainParams:
    epochs : int = 4
    steps_per_epoch : int = 1024
    visualize : bool = False
    env_params : DistributedEnvParams = DistributedEnvParams()

def trainer(params):
    
    # define the model
    # This part uses the layers and conventions from mechagogue.
    # In this setup, each layer is an "init" function and a "model" function
    # where the init function builds the initial weights, and the model function
    # is a forward pass.
    # The model below has three "layers."  The first reshapes the input, the
    # second is a linear layer and the last samples an action based on the
    # logits provided by the model.
    # The "layer_sequence" then wraps these layers into two new functions, one
    # "init_model" which initializes all the layers, and another "model" which
    # does a forward pass of all the layers in sequence.
    init_model, model = layer_sequence((
        # the first layer reshape the (5,5) input to a (25,) vector
        (lambda : None, lambda x : x.reshape(-1)),
        # the second layer is a linear layer mapping 25 -> 4
        linear_layer(25, 4, dtype=FLOAT_DTYPE),
        # the last layer samples the 4-way categorical distribution produced by
        # the linear layer
        categorical_sampler_layer(),
    ))
    
    # init_population and model_population simply vectorize init_model and model
    def init_population(key):
        model_keys = jrng.split(key, params.env_params.num_agents)
        population_state = jax.vmap(init_model)(model_keys)
        return population_state
    init_population = jax.jit(init_population)
    
    def model_population(key, x, state):
        model_keys = jrng.split(key, params.env_params.num_agents)
        x = jax.vmap(model)(model_keys, x, state)
        return x
    model_population = jax.jit(model_population)
    
    def mutate_population(key, state):
        new_weights = state[1][0] + jrng.normal(
            key, state[1][0].shape, dtype=FLOAT_DTYPE)
        new_bias = None
        return [state[0], (new_weights, new_bias), state[2]]
    
    def train(key):
        
        # build the environment reset and step functions
        reset_env, step_env = distributed_env(params.env_params)
        
        # get initialization keys
        key, env_key, model_key = jrng.split(key, 3)
        
        # initialize the environment
        env_state, observation = reset_env(env_key)
        
        # initialize the population state
        population_state = init_population(model_key)
        
        # make a function that will train one epoch that we can jit compile
        def train_epoch(key, epoch, env_state, observation, population_state):
            
            # make a function that will do one environmental/training step
            def train_step(key_state_obs_pop, _):
                
                # unpack the input
                key, env_state, observation, population_state = (
                    key_state_obs_pop)
                
                # get keys for the model, environment and mutation
                key, model_key, env_key, mutate_key = jrng.split(key, 4)
                
                # use the model to sample an action for each agent
                action = model_population(
                    model_key, observation, population_state)
                
                # take an environment step to get the next state and observation
                env_state, observation = step_env(env_key, env_state, action)
                
                # mutate the model
                population_state = mutate_population(
                    mutate_key, population_state)
                
                # return
                return (key, env_state, observation, population_state), None
            
            # scan (fast jax for loop)
            key, step_key = jrng.split(key)
            (key, env_state, observation, population_state), _ = jax.lax.scan(
                train_step,
                (key, env_state, observation, population_state),
                None,
                length=params.steps_per_epoch,
            )
            
            return key, env_state, observation, population_state
        
        # jit compile the train_epoch function
        train_epoch = jax.jit(train_epoch)
            
        # iterate through the epochs
        for epoch in range(params.epochs):
            print(f'Epoch: {epoch}')
            key, env_state, observation, population_state = train_epoch(
                key, epoch, env_state, observation, population_state)
            
            # if we wanted to, we could save env_state and population_state to
            # a checkpoint file here, since this is outside the jit compiled
            # function
    
    return train

if __name__ == '__main__':
    params = TrainParams().from_commandline()
    params.override_descendants()
    
    train = trainer(params)
    key = jrng.key(1234)
    train(key)
