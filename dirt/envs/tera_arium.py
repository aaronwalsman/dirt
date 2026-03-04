import math
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jrng
#from jax.experimental import checkify

import chex

from mechagogue.tree import tree_getitem
from mechagogue.static import static_data, static_functions
from mechagogue.dp.poeg import make_poeg
from mechagogue.debug import conditional_print

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
from dirt.gridworld2d.grid import (
    read_grid_locations, set_grid_shape, grid_mean_to_sum, upsample_grid)
from dirt.gridworld2d.observations import first_person_view, noisy_sensor
import dirt.gridworld2d.spawn as spawn
from dirt.visualization.image import jax_to_image

@static_data
class TeraAriumParams:
    verbose : bool = True
    
    distributed : bool = False
    tile_dimensions : Tuple[int, int] = (1, 1)
    
    world_size : Tuple[int, int] = (1024, 1024)
    spatial_offset : Tuple[int, int] = (0,0)
    max_size : Tuple[int, int] = None
    
    landscape_seed : int = None
    bug_seed : int = None
    
    initial_players : int = 1024
    max_players : int = 16384
    
    rock_mode : str = 'fractal'
    rock_bias : float = 0
    
    include_rock : bool = True
    include_water : bool = True
    include_energy : bool = True
    include_biomass : bool = True
    include_wind : bool = True
    include_temperature : bool = True
    include_rain : bool = True
    include_light : bool = True
    include_audio : bool = True
    audio_channels : int = 8
    include_smell : bool = True
    smell_channels : int = 8
    
    include_expell_actions : bool = True
    include_violence : bool = True
    
    # observations
    max_view_distance : int = 5
    max_view_back_distance : int = 5
    max_view_width : int = 11
    include_compass : bool = True
    vision_includes_rgb : bool = True
    vision_includes_relative_altitude : bool = True
    
    # corrections
    auto_correct_biomass : bool = False
    biomass_correction_sites : int = 32
    biomass_correction_overshoot : float = 1
    
    # reporting
    report_bug_actions : bool = False
    report_bug_internals : bool = False
    report_bug_traits : bool = False
    report_object_grid : bool = False
    report_homicides : bool = False
    display_hit_map : bool = False
    
    landscape : LandscapeParams = LandscapeParams()
    bugs : BugParams = BugParams()

@static_data
class TeraAriumState:
    landscape : LandscapeState
    bugs : BugState
    bug_traits : BugTraits
    
    initial_biomass : jnp.array
    step_biomass_correction : jnp.array
    
    homicides : jnp.array = None
    attacks : jnp.array = None
    homicide_locations : jnp.array = None
    hit_map : jnp.array = None
    migration_src : jnp.array = None
    migration_dst : jnp.array = None
    
TeraAriumTraits = BugTraits

def _compute_tile_size(params: TeraAriumParams) -> Tuple[int, int]:
    h, w = params.world_size
    tr, tc = params.tile_dimensions
    assert h % tr == 0 and w % tc == 0
    return (h // tr, w // tc)

def make_tera_arium(
    params : TeraAriumParams = TeraAriumParams(),
    float_dtype=DEFAULT_FLOAT_DTYPE,
):
    params = params.override_descendants()
    extra_halos = None
    if params.distributed:
        movement_halo = int(math.ceil(params.bugs.max_movement))
        extra_halos = {
            "rock": movement_halo,
            "water": movement_halo,
        }
    landscape_params = params.landscape
    if params.distributed and landscape_params.max_size is None:
        tr, tc = params.tile_dimensions
        h, w = params.world_size
        landscape_params = landscape_params.replace(
            max_size=(h * tr, w * tc),
        )
    landscape = make_landscape(
        landscape_params,
        float_dtype=float_dtype,
        extra_halos=extra_halos,
        distributed=params.distributed,
        tile_dimensions=params.tile_dimensions,
    )
    bugs = make_bugs(
        params.bugs,
        distributed=params.distributed,
        tile_dimensions=params.tile_dimensions,
    )
    if params.distributed and params.tile_dimensions != (1, 1):
        migration_shape = (8, params.max_players)
    else:
        migration_shape = (0, params.max_players)
    empty_migration = jnp.full(migration_shape, -1, dtype=jnp.int32)
    bug_object_grid = bugs.object_grid
    
    if params.distributed:
        _tile_size = _compute_tile_size(params)
    
    def init_state(
        key : chex.PRNGKey,
    ) -> TeraAriumState :
        
        if params.landscape_seed is None:
            if params.distributed:
                base_key = jax.lax.pmin(jrng.key_data(key), axis_name="mesh")
                landscape_key = jrng.wrap_key_data(base_key)
            else:
                key, landscape_key = jrng.split(key)
        else:
            landscape_key = jrng.key(params.landscape_seed)
        landscape_state = landscape.init(landscape_key)
        
        if params.bug_seed is None:
            key, bug_key = jrng.split(key)
        else:
            bug_key = jrng.key(params.bug_seed)
        bug_state = bugs.init(bug_key)
        bug_traits = BugTraits.default(params.max_players)
        
        initial_biomass = (
            jnp.sum(landscape_state.biomass, dtype=jnp.float32) +
            jnp.sum(bug_state.biomass, dtype=jnp.float32)
        )
        
        hit_map = jnp.zeros(params.world_size, dtype=float_dtype)
        homicides = jnp.zeros(params.max_players, dtype=jnp.bool)
        attacks = jnp.zeros(params.max_players, dtype=jnp.bool)
        homicide_locations = jnp.zeros((params.max_players, 2), dtype=jnp.int32)
        
        state = TeraAriumState(
            landscape_state,
            bug_state,
            bug_traits,
            initial_biomass=initial_biomass,
            step_biomass_correction=jnp.zeros((), dtype=jnp.float32),
            hit_map=hit_map,
            homicides=homicides,
            attacks=attacks,
            homicide_locations=homicide_locations,
            migration_src=empty_migration,
            migration_dst=empty_migration,
        )

        if params.distributed:
            state = state.replace(landscape=landscape.exchange(state.landscape))
            state = state.replace(bugs=bugs.exchange(state.bugs))
        
        return state
    
    def transition(
        key : chex.PRNGKey,
        state : TeraAriumState,
        action : int,
        traits : BugTraits,
    ) -> TeraAriumState :
        
        # bugs
        bug_state = state.bugs

        # TODO(halo): two-exchange plan.
        # 1) Pre-transition halos are assumed valid (from init or prior step).
        #    Transition can read grid values safely at this point.
        # 2) After local writes but BEFORE any neighbor-dependent dynamics
        #    (notably landscape.step), exchange halos again so dynamics see
        #    correct neighbor values.
        # 3) After transition, exchange halos again so observation is correct.
        
        # - eat
        #   do this before anything else happens so that the food an agent
        #   observed in the last time step is still in the right location
        # TODO(halo): if running distributed, ensure landscape grids have
        # up-to-date halo padding before any grid reads based on bug positions.
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
        # TODO(halo): photosynthesis reads light at bug locations; requires
        # light halo to be current near tile boundaries.
        if params.include_light:
            bug_light = landscape.get_light(state.landscape, bug_state.x)
        else:
            bug_light = jnp.ones((params.max_players,), dtype=float_dtype) 
        bug_state = bugs.photosynthesis(bug_state, traits, bug_light)
        
        # - heal
        bug_state = bugs.heal(bug_state, traits)
        
        # - metabolism
        bug_state, evaporated_metabolism = bugs.metabolism(bug_state, traits)
        
        # - fight
        # TODO(halo): bug collision/interaction should use an object_grid
        # with a fresh halo exchange before fight near tile edges.
        if params.distributed:
            state = state.replace(bugs=bugs.exchange(state.bugs))
            bug_state = state.bugs
        key, fight_key = jrng.split(key)
        bug_state, attacks, homicides, homicide_locations, hit_map = bugs.fight(
            fight_key, bug_state, action, traits)
        
        # - move bugs
        # TODO(halo): movement and collision checks should use an object_grid
        # with a fresh halo exchange before moving near tile edges.
        key, move_key = jrng.split(key)
        altitude = landscape.get_altitude_full(landscape_state)
        bug_state, evaporated_move, migrations = bugs.move_and_migrate(
            move_key,
            bug_state,
            action,
            traits,
            altitude,
            params.landscape.terrain_downsample,
            altitude_grid=landscape.altitude_grid(),
        )
        if migrations is not None:
            migration_src, migration_dst = migrations
        else:
            migration_src = empty_migration
            migration_dst = empty_migration
        
        # - birth and death
        # TODO(halo): birth/death may write to object_grid; consider halo
        # update after local writes if subsequent steps read neighbors.
        (
            bug_state,
            expelled_x,
            evaporated_birth,
            expelled_water,
            expelled_energy,
            expelled_biomass,
        ) = bugs.birth_and_death(bug_state, action, traits)
        
        # - add evaporated water to the atmosphere/ground
        # TODO(halo): add_* writes should remain interior-only; halo should be
        # refreshed before any neighbor-dependent reads (e.g., step/observe).
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

        # TODO(halo): exchange landscape halos HERE (after local writes) before
        # landscape.step. Grids likely needed:
        # - rock (if used in step), water, energy, biomass
        # - moisture/raining (if include_rain), temperature (if include_temperature)
        # - wind (if include_wind), light (if include_light)
        # - audio/smell (if include_audio/include_smell), any gas fields
        # This should use fill_value=0 for floats.
        
        # natural landscape processes
        # TODO(halo): landscape.step may depend on neighboring cells; ensure
        # required grids have halo padding before step.
        if params.distributed:
            state = state.replace(landscape=landscape_state)
            state = state.replace(landscape=landscape.exchange(state.landscape))
            landscape_state = state.landscape
        key, landscape_key = jrng.split(key)
        landscape_state = landscape.step(
            landscape_key, landscape_state)
        
        # fix biomass
        '''
        site_biomass = (
            state.step_biomass_correction *
            params.biomass_correction_overshoot /
            params.biomass_correction_sites
        )
        key, biomass_key = jrng.split(key)
        biomass_sites = spawn.uniform_x(
            biomass_key,
            params.biomass_correction_sites,
            params.landscape.resource_downsample,
        )
        corrected_biomass = landscape_state.biomass.at[
            biomass_sites[:,0], biomass_sites[:,1]].add(site_biomass)
        landscape_state = landscape_state.replace(biomass=corrected_biomass)
        '''
        
        state = state.replace(
            landscape=landscape_state,
            bugs=bug_state,
            bug_traits=traits,
            attacks=attacks,
            homicides=homicides,
            homicide_locations=homicide_locations,
            migration_src=migration_src,
            migration_dst=migration_dst,
        )
        if params.display_hit_map:
            state = state.replace(hit_map=hit_map)

        # TODO(halo): exchange halos HERE (end of transition) so observe sees
        # correct neighbor data. Include same landscape grids as above, plus
        # bug object_grid if observations or debug views depend on it.
        if params.distributed:
            state = state.replace(landscape=landscape.exchange(state.landscape))
            state = state.replace(bugs=bugs.exchange(state.bugs))
        
        return state
    
    def observe(
        key : chex.PRNGKey,
        state : TeraAriumState,
    ) -> BugObservation:
        # TODO(halo): observation expects halos to already be current (end of
        # transition or post-init exchange).
        # TODO(perf): investigate batching or fusing repeated grid reads
        # (read_grid_locations) across channels/fields.
        # visual
        # - rgb
        # TODO(halo): render_rgb and first_person_view should operate on grids
        # with current halos; use local coordinates if distributed.
        # TODO(perf): render_rgb may be expensive at full resolution; consider
        # a cheaper observation path if possible.
        if params.vision_includes_rgb:
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
        else:
            rgb_view = None
        
        # - relative altitude
        # TODO(halo): altitude reads and views need halos near tile edges.
        if params.vision_includes_relative_altitude:
            '''
            if params.include_rock:
                altitude = state.landscape.rock.copy()
                #altitude = jnp.zeros_like(state.landscape.rock)
                #altitude = jnp.ones((128,128), dtype=float_dtype)
                #slope = jnp.arange(0, 128, dtype=float_dtype)
                #slope = slope - 64
                #slope = slope * 4
                #altitude = altitude * slope * 0.1
                #altitude = grid_mean_to_sum(altitude, 4)
            else:
                altitude = jnp.zeros((1,1), dtype=float_dtype)
            if params.include_water:
                altitude += state.landscape.water
            '''
            
            altitude = landscape.get_altitude(state.landscape)
            
            bug_altitude = landscape.altitude_grid.read(
                landscape.get_altitude_full(state.landscape), state.bugs.x)
            altitude_view = first_person_view(
                state.bugs.x,
                state.bugs.r,
                landscape.altitude_grid.interior(altitude),
                params.max_view_width,
                params.max_view_distance,
                params.max_view_back_distance,
                downsample=params.landscape.terrain_downsample,
            )
            altitude_view = altitude_view - bug_altitude[:,None,None]
        else:
            altitude_view = None
        
        # audio/smell
        # TODO(halo): audio/smell reads need halos near tile edges.
        if params.include_audio:
            audio = landscape.get_audio(state.landscape, state.bugs.x)
        else:
            audio = None
        if params.include_smell:
            smell = landscape.get_smell(state.landscape, state.bugs.x)
        else:
            smell = None
        
        # weather
        # TODO(halo): wind/temperature reads need halos near tile edges.
        if params.include_wind:
            wind = state.landscape.wind / state.landscape.max_wind
            # TODO(perf): avoid per-player repeat if bugs.observe can accept
            # broadcasted/global wind directly.
            wind = jnp.repeat(
                wind[None,...], repeats=params.max_players, axis=0)
        else:
            wind = None
        if params.include_temperature:
            temperature = landscape.get_temperature(state.landscape, state.bugs.x)
        else:
            temperature = 0.
        
        # external resources
        # TODO(halo): resource reads need halos near tile edges.
        if params.include_water:
            external_water = landscape.get_water(state.landscape, state.bugs.x)
        else:
            external_water = None
        if params.include_energy:
            external_energy = landscape.get_energy(state.landscape, state.bugs.x)
        else:
            external_energy = None
        if params.include_biomass:
            external_biomass = landscape.get_biomass(state.landscape, state.bugs.x)
        else:
            external_biomass = None
        
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
    
    def family_info(state, action=None, next_state=None):
        if next_state is None:
            next_state = state
        return bugs.family_info(next_state.bugs)

    def migrations(state, action, next_state):
        return (next_state.migration_src, next_state.migration_dst)
    
    def correct(state, steps):
        if params.auto_correct_biomass:
            # TODO(perf): full-grid sums every call can be expensive; consider
            # tracking deltas or reducing correction frequency.
            #state_biomass = (
            #    jnp.sum(state.landscape.biomass, dtype=jnp.float32) +
            #    jnp.sum(state.bugs.biomass, dtype=jnp.float32)
            #)
            state_landscape_biomass = jnp.sum(
                landscape.biomass_grid.interior(state.landscape.biomass),
                dtype=jnp.float32,
            )
            state_bugs_biomass = jnp.sum(
                state.bugs.biomass, dtype=jnp.float32)
            state_biomass = state_landscape_biomass + state_bugs_biomass
            
            missing_biomass = jnp.clip(
                state.initial_biomass - state_biomass, min=0.)
            #step_biomass_correction = (
            #    missing_biomass / steps).astype(float_dtype)
            step_biomass_correction = jnp.where(
                missing_biomass > 0., params.biomass_correction_sites, 0.)
            state = state.replace(
                step_biomass_correction=step_biomass_correction)
            
            #target_landscape_biomass = state_biomass - next_state_bugs_biomass
            #loss_ratio = next_state_landscape_biomass / target_landscape_biomass
            #step_biomass_loss_ratio = loss_ratio ** (20./params.steps_per_epoch)
            #step_biomass_correcton = (
            #    1. / step_biomass_loss_ratio).astype(float_dtype)
            #next_state = next_state.replace(
            #    step_biomass_correction = step_biomass_correction)
        
        return state
    
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
        elif display_mode == 6 and params.report_object_grid:
            object_grid = bug_object_grid.interior(report.object_grid)
            occupied = object_grid != -1
            rgb = jnp.stack((occupied, occupied, occupied), axis=-1)
            rgb = set_grid_shape(rgb, *shape, preserve_mass=False)
            return jax_to_image(rgb)
        else:
            rgb = jnp.zeros((*shape, 3), dtype=float_dtype)
            return jax_to_image(rgb)
    
    def make_video_report(state):
        report = (landscape.render_rgb(
            state.landscape,
            params.world_size,
            spot_x = state.bugs.x,
            spot_color = state.bug_traits.color,
            use_light=False,
        ) * 255).astype(jnp.uint8)
        
        if params.display_hit_map:
            hit_mask = state.hit_map[...,None] > 0.
            report = jnp.where(
                hit_mask,
                jnp.array([255,255,255], dtype=jnp.uint8),
                report,
            )
            existing_colors = report[
                state.homicide_locations[...,0],
                state.homicide_locations[...,1],
            ]
            spot_color = jnp.where(
                state.homicides[...,None],
                jnp.array([255,0,0], dtype=jnp.uint8),
                existing_colors,
            )
            report = report.at[
                state.homicide_locations[...,0],
                state.homicide_locations[...,1],
            ].set(spot_color) 
        
        return report
    
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
            generation : jnp.ndarray = False
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
            player_color=state.bug_traits.color,
        )
        if params.include_rock:
            report = report.replace(
                rock=state.landscape.rock)
        if params.include_water:
            report = report.replace(
                water=state.landscape.water)
        if params.include_light:
            report = report.replace(
                light=state.landscape.light)
        if params.include_temperature:
            report = report.replace(
                temperature=state.landscape.temperature)
        if params.include_rain:
            report = report.replace(
                moisture=state.landscape.moisture,
                raining=state.landscape.raining,
            )
        if params.include_energy:
            report = report.replace(
                energy=state.landscape.energy)
        if params.include_biomass:
            report = report.replace(biomass=state.landscape.biomass)
        
        if params.report_bug_actions:
            report = report.replace(actions=actions)
        if params.report_bug_internals:
            report = report.replace(age=state.bugs.age)
            report = report.replace(generation=state.bugs.generation)
            report = report.replace(hp=state.bugs.hp)
            if params.include_water:
                report = report.replace(player_water=state.bugs.water)
            if params.include_energy:
                report = report.replace(player_energy=state.bugs.energy)
            if params.include_biomass:
                report = report.replace(player_biomass=state.bugs.biomass)
        
        if params.report_bug_traits:
            report = report.replace(traits=state.bug_traits)
        
        if params.report_object_grid:
            report = report.replace(
                object_grid=bug_object_grid.interior(state.bugs.object_grid))
        
        return report
    
    def print_player_info(player_id, report):
        print(f'ID:        {player_id}')
        if hasattr(report, "player_x"):
            print(f'  pos:        {report.player_x[player_id]}')
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
            print(f'  age:         {report.age[player_id]}')
            print(f'  generation:  {report.generation[player_id]}')
            print(f'  hp:          {report.hp[player_id]}')
            if params.include_water:
                print(f'  water:       {report.player_water[player_id]}')
            if params.include_energy:
                print(f'  energy:      {report.player_energy[player_id]}')
            if params.include_biomass:
                print(f'  biomass:     {report.player_biomass[player_id]}')
        
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
        normal_mutate_trait=bugs.normal_mutate_trait,
        empty_family_tree_state=bugs.empty_family_tree_state,
        visualizer_terrain_map=landscape.visualizer_terrain_map,
        visualizer_terrain_texture=visualizer_terrain_texture,
        default_visualizer_report=default_visualizer_report,
        make_video_report=make_video_report,
        make_visualizer_report=make_visualizer_report,
        print_player_info=print_player_info,
        num_actions=bugs.num_actions,
        action_primitive_count=bugs.action_primitive_count,
        action_to_primitive=bugs.action_to_primitive,
        biomass_requirement=bugs.biomass_requirement,
        correct=correct,
        capacity_reached=bugs.capacity_reached,
        migrations=migrations,
        distributed=params.distributed,
        tile_dimensions=params.tile_dimensions,
    )
    
    return game
