from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.tree import tree_getitem
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
    action_type_names,
)
from dirt.gridworld2d.grid import read_grid_locations, set_grid_shape
from dirt.gridworld2d.observations import first_person_view, noisy_sensor
from dirt.visualization.image import jax_to_image

@static_data
class TeraAriumParams:
    world_size : Tuple[int, int] = (1024, 1024)
    
    initial_players : int = 1024
    max_players : int = 16384
    
    include_rock : bool = True
    include_water : bool = True
    include_energy : bool = True
    include_biomass : bool = True
    include_temperature : bool = True
    include_rain : bool = True
    include_light : bool = True
    include_energy : bool = True
    include_biomass : bool = True
    
    # observations
    max_view_distance : int = 5
    max_view_back_distance : int = 5
    max_view_width : int = 11
    
    # reporting
    report_bug_actions : bool = False
    report_bug_internals : bool = False
    report_bug_traits : bool = False
    
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
    bugs = make_bugs(params.bugs)
    
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
            external_water=bug_water,
            external_energy=bug_energy,
            external_biomass=bug_biomass,
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
        if params.include_light:
            bug_light = grid.read_grid_locations(
                state.landscape.light,
                bug_state.x,
                params.landscape.light_downsample,
            )
        else:
            bug_light = jnp.ones((params.max_players,), dtype=float_dtype) 
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
        if params.include_water:
            evaporated_moisture = (
                evaporated_move + evaporated_metabolism + evaporated_birth)
            if params.landscape.include_rain:
                landscape_state = landscape.add_moisture(
                    landscape_state, expelled_x, evaporated_moisture)
            else:
                landscape_state = landscape.add_water(
                    landscape_state, expelled_x, evaporated_moisture)
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
            spot_color=state.bug_traits.color,
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
        if params.include_rock:
            altitude = state.landscape.rock
        else:
            altitude = jnp.zeros((1,1), dtype=float_dtype)
        if params.include_water:
            altitude += state.landscape.water
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
        if params.include_temperature:
            temperature = read_grid_locations(
                state.landscape.temperature,
                state.bugs.x,
                params.landscape.temperature_downsample,
            )
        else:
            temperature = 0.
        
        # external resources
        if params.include_water:
            '''
            external_water = read_grid_locations(
                state.landscape.water,
                state.bugs.x,
                params.landscape.terrain_downsample,
            )
            '''
            external_water = landscape.get_water(state.landscape, state.bugs.x)
        else:
            external_water = None
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
    
    def visualizer_terrain_texture(report, shape, display_mode):
        if display_mode in (1,2,3,4,5):
            return landscape.render_display_mode(
                report,
                shape,
                display_mode,
                spot_x=report.player_x,
                spot_color=report.player_color,
                convert_to_image=True,
            )
        elif display_mode == 6:
            occupied = report.object_grid != -1
            rgb = jnp.stack((occupied, occupied, occupied), axis=-1)
            rgb = set_grid_shape(rgb, *shape, preserve_mass=False)
            return jax_to_image(rgb)
    
    @static_data
    class VisualizerReport:
        if params.include_rock:
            rock : jnp.ndarray = False
        if params.include_water:
            water : jnp.ndarray = False
        if params.include_light:
            light : jnp.ndarray = False
        if params.include_temperature:
            temperature : jnp.ndarray = False
        if params.include_rain:
            moisture : jnp.ndarray = False
            raining : jnp.ndarray = False
        if params.include_energy:
            energy : jnp.ndarray = False
        if params.include_biomass:
            biomass : jnp.ndarray = False
     
        players : jnp.ndarray = False
        player_x : jnp.ndarray = False
        player_r : jnp.ndarray = False 
        object_grid : jnp.ndarray = False
        player_color : jnp.ndarray = False
        
        if params.report_bug_actions:
            actions : jnp.ndarray = False
        if params.report_bug_internals:
            age : jnp.ndarray = False
            hp : jnp.ndarray = False
            if params.include_water:
                player_water : jnp.ndarray = False
            if params.include_energy:
                player_energy : jnp.ndarray = False
            if params.include_biomass:
                player_biomass : jnp.ndarray = False
        if params.report_bug_traits:
            traits : BugTraits = BugTraits.default(())
    
    def default_visualizer_report():
        return VisualizerReport()
    
    def make_visualizer_report(state, actions):
        report = VisualizerReport(
            players=active_players(state),
            player_x=state.bugs.x,
            player_r=state.bugs.r,
            object_grid=state.bugs.object_grid,
            player_color=state.bug_traits.color,
        )
        if params.include_rock:
            report = report.replace(rock=state.landscape.rock)
        if params.include_water:
            report = report.replace(water=state.landscape.water)
        if params.include_light:
            report = report.replace(light=state.landscape.light)
        if params.include_temperature:
            report = report.replace(temperature=state.landscape.temperature)
        if params.include_rain:
            report = report.replace(
                moisture=state.landscape.moisture,
                raining=state.landscape.raining,
            )
        if params.include_energy:
            report = report.replace(energy=state.landscape.energy)
        if params.include_biomass:
            report = report.replace(biomass=state.landscape.biomass)
        
        if params.report_bug_actions:
            report = report.replace(actions=actions)
        if params.report_bug_internals:
            report = report.replace(age=state.bugs.age)
            report = report.replace(hp=state.bugs.hp)
            if params.include_water:
                report = report.replace(player_water=state.bugs.water)
            if params.include_energy:
                report = report.replace(player_energy=state.bugs.energy)
            if params.include_biomass:
                report = report.replace(player_biomass=state.bugs.biomass)
        
        if params.report_bug_traits:
            report = report.replace(traits=state.bug_traits)
        
        return report
    
    def print_player_info(player_id, report):
        print(f'ID:        {player_id}')
        if params.report_bug_actions:
            action_type, action_primitive = bugs.get_action_type_and_primitive(
                report.actions[player_id]) 
            print(
                f'  actions: '
                f'{action_type_names[int(action_type)]} '
                f'{action_primitive} '
                f'({report.actions[player_id]})'
            )
        if params.report_bug_internals:
            print(f'  age:     {report.age[player_id]}')
            print(f'  hp:      {report.hp[player_id]}')
            if params.include_water:
                print(f'  water:   {report.player_water[player_id]}')
            if params.include_energy:
                print(f'  energy:  {report.player_energy[player_id]}')
            if params.include_biomass:
                print(f'  biomass: {report.player_biomass[player_id]}')
        
        if params.report_bug_traits:
            bug_traits = tree_getitem(report.traits, player_id)
            for key, value in bug_traits.__dict__.items():
                if not callable(value):
                    print(f'  {key}: {value}')
    
    game = make_poeg(
        init_state,
        transition,
        observe,
        active_players,
        family_info,
        mutate_traits=bugs.mutate_traits,
        visualizer_terrain_map=landscape.visualizer_terrain_map,
        visualizer_terrain_texture=visualizer_terrain_texture,
        default_visualizer_report=default_visualizer_report,
        make_visualizer_report=make_visualizer_report,
        print_player_info=print_player_info,
        num_actions=bugs.num_actions,
    )
    
    return game
