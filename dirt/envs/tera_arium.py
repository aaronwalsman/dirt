from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.static import static_data, static_functions
from mechagogue.dp.poeg import make_poeg

from dirt.constants import (
    ROCK_COLOR,
    WATER_COLOR,
    ICE_COLOR,
    ENERGY_TINT,
    BIOMASS_TINT,
    BIOMASS_AND_ENERGY_TINT,
    DEFAULT_FLOAT_DTYPE,
)
import dirt.gridworld2d.grid as grid
from dirt.gridworld2d.landscape import (
    LandscapeParams,
    LandscapeState,
    make_landscape,
)
from dirt.bug import (
    BugParams,
    BugTraits,
    BugObservation,
    BugState,
    make_bugs,
)
from dirt.gridworld2d.grid import read_grid_locations
from dirt.gridworld2d.observations import first_person_view, noisy_sensor

@static_data
class TeraAriumParams:
    world_size : Tuple[int, int] = (1024, 1024)
    
    initial_players : int = 1024
    max_players : int = 16384
    
    include_water : bool = True
    include_energy : bool = True
    include_biomass : bool = True
    
    # observations
    max_view_width : int = 11
    max_view_distance : int = 5
    max_view_back_distance : int = 5
    
    landscape : LandscapeParams = LandscapeParams()
    bugs : BugParams = BugParams()

@static_data
class TeraAriumState:
    landscape : LandscapeState
    bugs : BugState
    bug_traits : BugTraits

# @static_data
# class TeraAriumObservation:
#     # grid external
#     rgb : jnp.ndarray
#     relative_altitude : jnp.ndarray
#     
#     # single channel external
#     # - sensory
#     audio : jnp.ndarray
#     smell : jnp.ndarray
#     # - resources
#     external_water : jnp.ndarray
#     external_energy : jnp.ndarray
#     external_biomass : jnp.ndarray
#     wind : jnp.ndarray
#     temperature : jnp.ndarray
#     
#     # single channel internal   
#     health : jnp.ndarray
#     internal_water : jnp.ndarray
#     internal_energy : jnp.ndarray
#     internal_biomass : jnp.ndarray

#TeraAriumAction = BugAction
TeraAriumTraits = BugTraits

def make_tera_arium(
    params : TeraAriumParams = TeraAriumParams(),
    float_dtype=DEFAULT_FLOAT_DTYPE,
):
    
    landscape = make_landscape(params.landscape, float_dtype=float_dtype)
    bugs = make_bugs(params.bugs, float_dtype=float_dtype)
    
    def init_state(
        key : chex.PRNGKey,
    ) -> TeraAriumState :
        
        key, landscape_key = jrng.split(key)
        landscape_state = landscape.init(landscape_key)
        
        key, bug_key = jrng.split(key)
        bug_state = bugs.init(bug_key)
        bug_traits = BugTraits.default(params.max_players)
        
        state = TeraAriumState(landscape_state, bug_state, bug_traits)
        
        return state
    
    def transition(
        key : chex.PRNGKey,
        state : TeraAriumState,
        action : int,
        traits : BugTraits,
    ) -> TeraAriumState :
        
        # bugs
        bug_state = state.bugs
        
        # - eat
        #   do this before anything else happens so that the food an agent
        #   observed in the last time step is still in the right location
        # -- pull resources out of the environment
        landscape_state = state.landscape
        if params.include_water:
            landscape_state, bug_water = landscape.take_water(
                landscape_state, bug_state.x)
        else:
            bug_water = None
        if params.include_energy:
            landscape_state, bug_energy = landscape.take_energy(
                landscape_state, bug_state.x)
        else:
            bug_energy = None
        if params.include_biomass:
            landscape_state, bug_biomass = landscape.take_biomass(
                landscape_state, bug_state.x)
        else:
            bug_biomass = None
        # -- feed the resources to the bugs
        bug_state, leftover_water, leftover_energy, leftover_biomass = bugs.eat(
            bug_state,
            action,
            traits,
            water=bug_water,
            energy=bug_energy,
            biomass=bug_biomass,
        )
        # -- put the leftovers back in the environment
        if params.include_water:
            landscape_state = landscape.add_water(
                landscape_state, bug_state.x, leftover_water)
        if params.include_energy:
            landscape_state = landscape.add_energy(
                landscape_state, bug_state.x, leftover_energy)
        if params.include_biomass:
            landscape_state = landscape.add_biomass(
                landscape_state, bug_state.x, leftover_biomass)
        
        # - photosynthesis
        bug_light = grid.read_grid_locations(
            state.landscape.light,
            bug_state.x,
            params.landscape.light_downsample,
        )
        bug_state = bugs.photosynthesis(bug_state, traits, bug_light)
        
        # - heal
        bug_state = bugs.heal(bug_state, traits)
        
        # - metabolism
        bug_state, evaporated_metabolism = bugs.metabolism(bug_state, traits)
        
        # - fight
        # bug_state = bugs.fight(bug_state, action, traits)
        
        # - move bugs
        key, move_key = jrng.split(key)
        altitude = landscape.get_altitude(landscape_state)
        bug_state, evaporated_move = bugs.move(
            move_key,
            bug_state,
            action,
            traits,
            altitude,
            params.landscape.terrain_downsample,
        )
        
        # - birth and death
        (
            bug_state,
            expelled_x,
            evaporated_birth,
            expelled_water,
            expelled_energy,
            expelled_biomass,
        ) = bugs.birth_and_death(bug_state, action, traits)
        
        # - add evaporated water to the atmosphere/ground
        evaporated_moisture = (
            evaporated_move + evaporated_metabolism + evaporated_birth)
        if params.landscape.include_rain:
            landscape_state = landscape.add_moisture(
                landscape_state, expelled_x, evaporated_moisture)
        elif params.include_water:
            landscape_state = landscape.add_water(
                landscape_state, expelled_x, evaporated_moisture)
        if params.include_water:
            landscape_state = landscape.add_water(
                landscape_state, expelled_x, expelled_water)
        if params.include_energy:
            landscape_state = landscape.add_energy(
                landscape_state, expelled_x, expelled_energy)
        if params.include_biomass:
            landscape_state = landscape.add_biomass(
                landscape_state, expelled_x, expelled_biomass)
        
        # natural landscape processes
        key, landscape_key = jrng.split(key)
        landscape_state = landscape.step(
            landscape_key, landscape_state)
        
        state = state.replace(
            landscape=landscape_state,
            bugs=bug_state,
            bug_traits=traits,
        )
        
        return state
    
    def observe(
        key : chex.PRNGKey,
        state : TeraAriumState,
    ) -> BugObservation:
        # visual
        # - rgb
        rgb = landscape.render_rgb(
            state.landscape,
            params.world_size,
            spot_x=state.bugs.x,
            spot_color=state.bugs.color,
        )
        rgb_view = first_person_view(
            state.bugs.x,
            state.bugs.r,
            rgb,
            params.max_view_width,
            params.max_view_distance,
            params.max_view_back_distance,
        )
        
        # - relative altitude
        altitude = state.landscape.rock + state.landscape.water
        bug_altitude = read_grid_locations(
            altitude, state.bugs.x, params.landscape.terrain_downsample)
        altitude_view = first_person_view(
            state.bugs.x,
            state.bugs.r,
            altitude,
            params.max_view_width,
            params.max_view_distance,
            params.max_view_back_distance,
            downsample=params.landscape.terrain_downsample,
        )
        altitude_view = altitude_view - bug_altitude[:,None,None]
        
        # audio/smell
        audio = read_grid_locations(
            state.landscape.audio,
            state.bugs.x,
            params.landscape.audio_downsample,
        )
        smell = read_grid_locations(
            state.landscape.smell,
            state.bugs.x,
            params.landscape.smell_downsample,
        )
        
        # weather
        wind = state.landscape.wind / state.landscape.max_wind
        wind = jnp.repeat(wind[None,...], repeats=params.max_players, axis=0)
        temperature = read_grid_locations(
            state.landscape.temperature,
            state.bugs.x,
            params.landscape.temperature_downsample,
        )
        
        # external resources
        external_water = read_grid_locations(
            state.landscape.water,
            state.bugs.x,
            params.landscape.terrain_downsample,
        )
        external_energy = read_grid_locations(
            state.landscape.energy,
            state.bugs.x,
            params.landscape.resource_downsample,
        )
        external_biomass = read_grid_locations(
            state.landscape.biomass,
            state.bugs.x,
            params.landscape.resource_downsample,
        )
        
        return bugs.observe(
            key,
            state.bugs,
            state.bug_traits,
            rgb_view,
            altitude_view,
            audio,
            smell,
            wind,
            temperature,
            external_water,
            external_energy,
            external_biomass,
        )
    
    def active_players(state):
        return bugs.active_players(state.bugs)
    
    def family_info(state):
        return bugs.family_info(state.bugs)
    
    '''
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
        # # apply the energy tint
        # rgb = rgb + clipped_energy[..., None] * ENERGY_TINT

        # # apply the biomass tint
        # rgb = rgb + clipped_biomass[..., None] * BIOMASS_TINT
        
        # overlay the bug colors
        rgb = rgb.at[bug_x[...,0], bug_x[...,1]].set(bug_color)
        
        # apply lighting
        rgb = rgb * light[..., None]
        
        # clip between 0 and 1
        rgb = jnp.clip(rgb, min=0., max=1.)
        
        return rgb
    '''
    
    def visualizer_terrain_texture(report, shape, display_mode):
        return landscape.render_display_mode(
            report,
            shape,
            display_mode,
            spot_x=report.player_x,
            spot_color=report.player_color,
            convert_to_image=True,
        )
    
    game = make_poeg(
        init_state,
        transition,
        observe,
        active_players,
        family_info,
        visualizer_terrain_map=landscape.visualizer_terrain_map,
        visualizer_terrain_texture=visualizer_terrain_texture,
    )
    
    game.num_actions = bugs.num_actions
    
    return game
