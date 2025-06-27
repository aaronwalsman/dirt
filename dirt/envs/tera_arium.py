from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.static import static_data
from mechagogue.dp.poeg import make_poeg

from dirt.constants import (
    ROCK_COLOR,
    WATER_COLOR,
    ICE_COLOR,
    ENERGY_TINT,
    BIOMASS_TINT,
    BIOMASS_AND_ENERGY_TINT,
)
from dirt.envs.landscape import (
    LandscapeParams,
    LandscapeState,
    make_landscape,
)
from dirt.bug import (
    BugParams,
    BugAction,
    BugTraits,
    BugState,
    make_bugs,
)
from dirt.consumable import Consumable

@static_data
class TeraAriumParams:
    world_size : Tuple[int, int] = (1024, 1024)
    
    initial_players : int = 1024
    max_players : int = 16384
    
    landscape : LandscapeParams = LandscapeParams()
    bugs : BugParams = BugParams()

@static_data
class TeraAriumState:
    landscape : LandscapeState
    bugs : BugState

@static_data
class TeraAriumObservation:
    # grid external
    rgb : jnp.ndarray
    height : jnp.ndarray
    
    # single channel external
    audio : jnp.ndarray
    smell : jnp.ndarray
    taste : Consumable
    moisture : jnp.ndarray
    wind : jnp.ndarray
    temperature : jnp.ndarray
    
    # single channel internal   
    health : jnp.ndarray
    water : jnp.ndarray
    energy : jnp.ndarray
    biomass : jnp.ndarray

TeraAriumAction = BugAction
TeraAriumTraits = BugTraits

def make_tera_arium(params : TeraAriumParams = TeraAriumParams()):
    
    landscape = make_landscape(params.landscape)
    bugs = make_bugs(params.bugs)
    
    def init(
        key : chex.PRNGKey,
        initial_players : int,
        parent_traits : BugTraits,
        bug_x : Optional[jnp.ndarray],
        bug_r : Optional[jnp.ndarray],
    ) -> TeraAriumState :
        
        key, landscape_key = jrng.split(key)
        landscape_state = landscape.init(landscape_key)
        
        key, bug_key = jrng.split(key)
        bug_state = init_bugs(
            bug_key, initial_players, parent_traits, x=bug_x, r=bug_r)
        
        state = TeraAriumState(landscape_state, bug_state)
        
        return state
    
    def observe(
        key : chex.PRNGKey,
        state : TeraAriumState,
    ) -> TeraAriumObservation:
        # TODO
        # player internal state
        
        # player external observation
        return None
    
    def transition(
        key : chex.PRNGKey,
        state : TeraAriumState,
        action : BugAction,
        traits : BugTraits,
    ) -> TeraAriumState :
        
        # bugs
        bug_state = state.bugs
        
        max_players = state.bugs.family_tree.parents.shape[0]
        deaths = jnp.zeros(max_players, dtype=jnp.bool)
        
        # - eat
        #   do this before anything else happens so that the food an agent
        #   observed in the last time step is still in the right location
        # -- pull resources out of the environment
        landscape_state = state.landscape
        landscape_state, bug_water = landscape.take_water(
            landscape_state, bug_state.x)
        landscape_state, bug_energy = landscape.take_energy(
            landscape_state, bug_state.x)
        landscape_state, bug_biomass = landscape.take_biomass(
            landscape_state, bug_state.x)
        # -- feed them to the bugs
        bug_state, leftover_water, leftover_energy, leftover_biomass = bugs.eat(
            bug_state, bug_water, bug_energy, bug_biomass)
        # -- put the leftovers back in the environment
        landscape_state = landscape.add_water(
            landscape_state, bug_state.x, leftover_water)
        landscape_state = landscape.add_energy(
            landscape_state, bug_state.x, leftover_energy)
        landscape_state = landscape.add_biomass(
            landscape_state, bug_state.x, leftover_biomass)
        
        # - fight
        pass
        
        # - move bugs
        next_bug_state = bugs.move(bug_state, action)
        
        # - metabolize and reproduce
        # TODO: does this need to be all in the same place?
        bug_state, expelled_water, expelled_energy, expelled_biomass, expelled_locations = bug_metabolism(
            bug_state,
            action,
            next_bug_state,
            traits,
            landscape_state.terrain,
            landscape_state.water,
            landscape_state.light,
        )
        # -- put the expelled resources back into the landscape
        #next_landscape_state = landscape.add_consumable(
        #    next_landscape_state, expelled_locations, expelled)
        
        # natural landscape processes
        key, landscape_key = jrng.split(key)
        next_landscape_state = landscape.step(
            landscape_key, next_landscape_state)
        
        next_state = next_state.replace(
            landscape=next_landscape_state,
            bugs=next_bug_state
        )
        
        return next_state
    
    def active_players(state):
        return active_bugs(state.bugs)
    
    def family_info(state):
        return bug_family_info(state.bugs)
    
    def render(
        water,
        temperature,
        energy,
        biomass,
        bug_x,
        bug_color,
        light,
    ):
        h, w = water.shape

        # start with a baseline rock color of 50% gray
        rgb = jnp.full((h,w,3), ROCK_COLOR, dtype=water.dtype)

        # overlay the water as blue and ice as white
        while len(temperature.shape) < 3:
            temperature = temperature[..., None]
        water_color = jnp.where(
            temperature <= 0,
            ICE_COLOR,
            WATER_COLOR,
        )
        rgb = jnp.where(water[..., None] > 0.05, water_color, rgb)
        
        # apply the energy and biomass tint
        clipped_energy = jnp.clip(energy, min=0., max=1.)
        clipped_biomass = jnp.clip(biomass, min=0., max=1.)
        biomass_and_energy = jnp.minimum(
            clipped_energy, clipped_biomass)
        just_energy = clipped_energy - biomass_and_energy
        just_biomass = clipped_biomass - biomass_and_energy
        rgb = rgb + biomass_and_energy[..., None] * BIOMASS_AND_ENERGY_TINT
        rgb = rgb + just_energy[..., None] * ENERGY_TINT
        rgb = rgb + just_biomass[..., None] * BIOMASS_TINT
        '''
        # apply the energy tint
        rgb = rgb + clipped_energy[..., None] * ENERGY_TINT

        # apply the biomass tint
        rgb = rgb + clipped_biomass[..., None] * BIOMASS_TINT
        '''
        
        # overlay the bug colors
        rgb = rgb.at[bug_x[...,0], bug_x[...,1]].set(bug_color)
        
        # apply lighting
        rgb = rgb * light[..., None]
        
        # clip between 0 and 1
        rgb = jnp.clip(rgb, min=0., max=1.)
        
        return rgb
    
    game = make_poeg(
        init_state, transition, observe, active_players, family_info)
    
    game.render = staticmethod(render)
    
    return game
