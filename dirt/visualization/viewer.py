import numpy as np

import jax
import jax.numpy as jnp

import glfw
import splendor.core as core
import splendor.contexts.glfw_context as glfw_context
from splendor.interactive_camera_glfw import InteractiveCameraGLFW
from splendor.frame_buffer import FrameBufferWrapper
import splendor.camera as camera
from splendor.masks import color_index_to_float, color_byte_to_index
import splendor.primitives as primitives
from splendor.image import save_image

from mechagogue.tree import tree_len, tree_getitem
from mechagogue.serial import load_example_data
from mechagogue.standardize import standardize_args
#from mechagogue.arg_wrappers import ignore_unused_args

from dirt.gridworld2d.grid import read_grid_locations
from dirt.visualization.height_map import (
    make_height_map_vertices_and_normals, make_height_map_mesh)

default_get_active_players = lambda report : report.players
default_get_player_x = lambda report : report.player_x
default_get_player_r = lambda report : report.player_r
default_get_terrain_map = lambda params : jnp.zeros(
    params.env_params.world_size)
#def default_get_terrain_map(params):
#    h, w = params.env_params.world_size
#    yz = jnp.sin(jnp.linspace(0, 3*2*jnp.pi, h))
#    xz = jnp.sin(jnp.linspace(0, 2*2*jnp.pi, w))
#    return yz[:,None] + xz[None,:]
#default_get_water_map = lambda : None
#default_get_player_color = lambda player_id : color_index_to_float(player_id+1)
def default_get_player_color(player_id, report):
    if hasattr(report, 'player_color'):
        return report.player_color[player_id]
    else:
        return color_index_to_float(player_id+1)

def default_print_player_info(player_id):
    print(player_id)

PLAYER_RADIUS=0.4

class Viewer:
    def __init__(
        self,
        #example_params,
        #params_file,
        example_report,
        report_files,
        world_size,
        window_size=(512,512),
        step_0 = 0,
        start_step=0,
        terrain_texture_resolution=None,
        max_render_players=512,
        get_report_block=lambda report : report,
        get_active_players=default_get_active_players,
        get_player_x=default_get_player_x,
        get_player_r=default_get_player_r,
        get_player_energy=None,
        get_terrain_map=default_get_terrain_map,
        get_terrain_texture=None,
        get_water_map=None,
        get_player_color=default_get_player_color,
        get_sun_direction=None,
        print_player_info=default_print_player_info,
    ):
        
        print('A')
        self.get_report_block = standardize_args(
            get_report_block, ('report',))
        if get_active_players is not None:
            self.get_active_players = standardize_args(
                #get_active_players, ('params', 'report'))
                get_active_players, ('report',))
        else:
            self.get_active_players = get_active_players
        self.get_player_x = standardize_args(
        #    get_player_x, ('params', 'report'))
            get_player_x, ('report',))
        self.get_player_r = standardize_args(
        #    get_player_r, ('params', 'report'))
            get_player_r, ('report',))
        self.get_player_energy = get_player_energy
        if get_player_energy:
            self.get_player_energy = standardize_args(
                #self.get_player_energy, ('params', 'report'))
                self.get_player_energy, ('report',))
        self.get_terrain_map = standardize_args(
            #get_terrain_map, ('params', 'report'))
            get_terrain_map, ('report',))
        self.get_terrain_texture = get_terrain_texture
        if self.get_terrain_texture:
            self.get_terrain_texture = standardize_args(
                self.get_terrain_texture,
                #('params', 'report', 'shape', 'display_mode'),
                ('report', 'shape', 'display_mode',),
            )
        self.get_water_map = get_water_map
        if self.get_water_map:
            self.get_water_map = standardize_args(
                #get_water_map, ('params', 'report'))
                get_water_map, ('report',))
        self.get_player_color = standardize_args(
            #get_player_color, ('player_id', 'params', 'report'))
            get_player_color, ('player_id', 'report'))
        if get_sun_direction:
            self.get_sun_direction = standardize_args(
                #get_sun_direction, ('params', 'report'))
                get_sun_direction, ('report',))
        if print_player_info:
            self.print_player_info = standardize_args(
                print_player_info, ('player_id', 'report',))
        
        self.world_size = world_size
        
        print('B')
        self._init_params_and_reports(
            #example_params,
            #params_file,
            example_report,
            report_files,
            step_0=step_0,
            start_step=start_step,
        )
        print('C')
        self._init_context_and_window(window_size[0], window_size[1])
        self._init_splendor_render()
        self._init_landscape(terrain_texture_resolution)
        if self.get_active_players is not None:
            self._init_players(max_render_players)
        self._init_camera_and_lights()
        self._init_callbacks()
        
        self.display_mode = 1
        self.show_players = True
        
        self.step_size = 1
        self.change_step(start_step)
    
    def _init_params_and_reports(
        self,
        #example_params,
        #params_file,
        example_report,
        report_files,
        step_0,
        start_step,
    ):
        self.step_0 = step_0
        self.current_step = start_step
        self.block_index = 0
        self._example_report = example_report
        
        #self.params = load_example_data(example_params, params_file)
        self.report_files = report_files
        report_block = load_example_data(
            self._example_report, self.report_files[self.block_index])
        self.current_report_block = self.get_report_block(report_block)
        self.report = tree_getitem(
            self.current_report_block, 0)
        self.reports_per_block = tree_len(self.current_report_block)
        
        self.step_N = step_0 + len(report_files) * self.reports_per_block
    
    def _init_context_and_window(self, window_width, window_height):
        glfw_context.initialize()
        self.window = glfw_context.GLFWWindowWrapper(
            width=window_width,
            height=window_height,
            anti_alias=False,
            anti_alias_samples=0,
        )
        #self._mask_framebuffer = FrameBufferWrapper(
        #    width=window_width,
        #    height=window_height,
        #    anti_alias=False,
        #)
    
    def _init_splendor_render(self):
        self.renderer = core.SplendorRender()
        self.upright = np.array([
            [ 1, 0, 0, 0],
            [ 0, 0, 1, 0],
            [ 0,-1, 0, 0],
            [ 0, 0, 0, 1],
        ])
    
    def _init_landscape(self, terrain_texture_resolution):
        #self.terrain_map = self.get_terrain_map(self.params, self.report)
        self.terrain_map = self.get_terrain_map(self.report)
        h, w = self.terrain_map.shape
        if terrain_texture_resolution is None:
            terrain_texture_resolution = self.world_size
        self.terrain_texture_resolution = terrain_texture_resolution
        self.mesh_spacing = self.world_size[0] / h
        
        vertices, normals, uvs, faces = make_height_map_mesh(
            self.terrain_map, spacing=self.mesh_spacing)
        self.terrain_uvs = uvs
        self.terrain_faces = faces
        self.renderer.load_mesh(
            name='terrain_mesh',
            mesh_data={
                'vertices' : vertices,
                'normals' : normals,
                'faces' : faces,
                'uvs' : uvs,
            },
            color_mode='textured',
        )
        
        self.renderer.load_texture(
            name='terrain_texture',
            texture_data=np.full(
                (self.terrain_texture_resolution + (3,)),
                127,
                dtype=np.uint8,
            ),
        )
        
        self.renderer.load_material(
            name='terrain_material',
            #flat_color=(0.5,0.5,0.5),
            texture_name='terrain_texture',
            rough=1.,
        )
        
        self.renderer.add_instance(
            name='terrain',
            mesh_name='terrain_mesh',
            material_name='terrain_material',
            transform=self.upright,
        )
        
        if self.get_water_map is not None:
            #water_map = self.get_water_map(self.params, self.report)
            water_map = self.get_water_map(self.report)
            self.total_height_map = self.terrain_map + water_map
            (
                water_vertices,
                water_normals,
                self.water_uvs,
                self.water_faces,
            ) = make_height_map_mesh(
                self.total_height_map, spacing=mesh_spacing)
            self.renderer.load_mesh(
                name='water_mesh',
                mesh_data={
                    'vertices' : water_vertices,
                    'normals' : water_normals,
                    'faces' : self.water_faces,
                    'uvs' : self.water_uvs,
                },
                color_mode='flat_color',
            )
            
            self.renderer.load_material(
                name='water_material',
                flat_color=(66/255.,135/255.,255.),
                rough=1.,
            )
            
            self.renderer.add_instance(
                name='water',
                mesh_name='water_mesh',
                material_name='water_material',
                transform=self.upright,
            )
        
        if hasattr(self, 'get_sun_direction'):
            max_size = max(self.world_size)
            self.renderer.load_mesh(
                name='sun_mesh',
                mesh_primitive={
                    'shape' : 'sphere',
                    'radius' : max_size * 0.05,
                },
                color_mode='flat_color',
            )
            self.renderer.load_material(
                name='sun_material',
                flat_color=(1,1,1),
                rough=1.,
            )
            self.renderer.add_instance(
                name='sun',
                mesh_name='sun_mesh',
                material_name='sun_material',
                transform = self.upright,
            )
            self._update_sun_position()
    
    def _update_sun_position(self):
        if hasattr(self, 'get_sun_direction'):
            #sun_direction = self.get_sun_direction(self.params, self.report)
            sun_direction = self.get_sun_direction(self.report)
            max_size = max(self.world_size)
            sun_transform = np.eye(4)
            sun_transform[:3,3] = sun_direction * max_size * 1
            #unclear_offset = jnp.array([
            #    [ 0, 1, 0, 0],
            #    [ 1, 0, 0, 0],
            #    [ 0, 0, 1, 0],
            #    [ 0, 0, 0, 1],
            #])
            sun_transform = sun_transform
            
            
            self.renderer.set_instance_transform('sun', sun_transform)
    
    def _init_players(self, max_render_players):
        #active_players = self.get_active_players(self.params, self.report)
        active_players = self.get_active_players(self.report)
        self.max_players = min(active_players.shape[0], max_render_players)
        self.selected_player = None
        
        # make player cube
        self.renderer.load_mesh(
            name='player_mesh',
            mesh_primitive={
                'shape':'cube',
                'x_extents':(-PLAYER_RADIUS, PLAYER_RADIUS),
                'y_extents':(-PLAYER_RADIUS, PLAYER_RADIUS),
                'z_extents':(-PLAYER_RADIUS, PLAYER_RADIUS),
                'bezel':0.15,
            },
            color_mode='flat_color',
        )
        
        # make player eye whites
        eye_white_mesh_a = primitives.disk(
            radius=0.15,
            inner_radius=0.05,
        )
        eye_white_mesh_a['vertices'][:,0] += 0.2
        eye_white_mesh_a['vertices'][:,1] += PLAYER_RADIUS + 0.01
        eye_white_mesh_a['vertices'][:,2] += 0.1
        
        eye_white_mesh_b = primitives.disk(
            radius=0.15,
            inner_radius=0.05,
        )
        eye_white_mesh_b['vertices'][:,0] -= 0.2
        eye_white_mesh_b['vertices'][:,1] += PLAYER_RADIUS + 0.01
        eye_white_mesh_b['vertices'][:,2] += 0.1
        
        eye_white_mesh = primitives.merge_meshes(
            [eye_white_mesh_a, eye_white_mesh_b])
        self.renderer.load_mesh(
            name='eye_white_mesh',
            mesh_data=eye_white_mesh,
            color_mode='flat_color',
        )
        
        self.renderer.load_material(
            name='eye_white_material',
            flat_color=(1.,1.,1.),
        )
        
        # make player eye pupils
        eye_pupil_mesh_a = primitives.disk(
            radius=0.05,
        )
        eye_pupil_mesh_a['vertices'][:,0] += 0.2
        eye_pupil_mesh_a['vertices'][:,1] += PLAYER_RADIUS + 0.01
        eye_pupil_mesh_a['vertices'][:,2] += 0.1
        
        eye_pupil_mesh_b = primitives.disk(
            radius=0.05,
        )
        eye_pupil_mesh_b['vertices'][:,0] -= 0.2
        eye_pupil_mesh_b['vertices'][:,1] += PLAYER_RADIUS + 0.01
        eye_pupil_mesh_b['vertices'][:,2] += 0.1
        
        eye_pupil_mesh = primitives.merge_meshes(
            [eye_pupil_mesh_a, eye_pupil_mesh_b])
        self.renderer.load_mesh(
            name='eye_pupil_mesh',
            mesh_data=eye_pupil_mesh,
            color_mode='flat_color',
        )
        
        self.renderer.load_material(
            name='eye_pupil_material',
            flat_color=(0.,0.,0.),
        )
        
        # make energy meters
        if self.get_player_energy is not None:
            self.renderer.load_mesh(
                name='energy_background_mesh',
                mesh_primitive={
                    'shape':'cube',
                    'x_extents':(-PLAYER_RADIUS, PLAYER_RADIUS),
                    'y_extents':(-0.05, 0.05),
                    'z_extents':(0, 0.4),
                },
                color_mode='flat_color',
            )
            self.renderer.load_material(
                name='energy_background_material',
                flat_color=(0.,0.,0.),
            )
            self.renderer.load_mesh(
                name='energy_mesh',
                mesh_primitive={
                    'shape':'cube',
                    'x_extents':(-PLAYER_RADIUS+0.05, PLAYER_RADIUS-0.05),
                    'y_extents':(-0.1, 0.1),
                    'z_extents':(0.05, 0.35),
                },
                color_mode='flat_color',
            )
            self.renderer.load_material(
                name='energy_material',
                flat_color=(0.,1.,0.),
            )
        
        self.player_instances = []
        self.player_eye_instances = []
        for player_id in range(self.max_players):
            material_name = f'player_material_{player_id}'
            self.renderer.load_material(
                name=material_name,
                flat_color=(0,0,0),
                rough=1.,
                metal=0.,
            )
            
            player_mask_color = color_index_to_float(player_id+1)
            
            player_name = f'player_{player_id}'
            self.renderer.add_instance(
                name=player_name,
                mesh_name='player_mesh',
                material_name=material_name,
                transform=np.eye(4),
                mask_color=player_mask_color,
                hidden=True,
            )
            
            player_eye_white_name = f'player_eye_white_{player_id}'
            self.renderer.add_instance(
                name=player_eye_white_name,
                mesh_name='eye_white_mesh',
                material_name='eye_white_material',
                transform=np.eye(4),
                mask_color=player_mask_color,
                hidden=True,
            )
            
            player_eye_pupil_name = f'player_eye_pupil_{player_id}'
            self.renderer.add_instance(
                name=player_eye_pupil_name,
                mesh_name='eye_pupil_mesh',
                material_name='eye_pupil_material',
                transform=np.eye(4),
                mask_color=player_mask_color,
                hidden=True,
            )
            
            if self.get_player_energy is not None:
                player_energy_background_name = (
                    f'player_energy_background_{player_id}')
                self.renderer.add_instance(
                    name=player_energy_background_name,
                    mesh_name='energy_background_mesh',
                    material_name='energy_background_material',
                    transform=np.eye(4),
                    hidden=True,
                )
                player_energy_name = f'player_energy_{player_id}'
                self.renderer.add_instance(
                    name=player_energy_name,
                    mesh_name='energy_mesh',
                    material_name='energy_material',
                    transform=np.eye(4),
                    hidden=True,
                )
    
    def _init_camera_and_lights(self):
        
        projection = camera.projection_matrix(
            np.radians(90.), 1., near_clip=1., far_clip=5000)
        self.renderer.set_projection(projection)
        
        c = np.cos(np.radians(-45.))
        s = np.sin(np.radians(-45.))
        d = max(self.world_size)
        camera_pose = np.array([
            [1, 0, 0, -0.5],
            [0, c,-s, d],
            [0, s, c, d],
            [0, 0, 0, 1],
        ])
        camera_pose = np.array([
            [ 0, 0,-1, 0],
            [ 0, 1, 0, 0],
            [ 1, 0, 0, 0],
            [ 0, 0, 0, 1],
        ]) @ camera_pose
        
        self.renderer.set_view_matrix(np.linalg.inv(camera_pose))
        
        self.camera_control = InteractiveCameraGLFW(self.window, self.renderer)
        
        '''
        self.renderer.load_cubemap(
            'grey_cube_dif',
            cubemap_asset='grey_cube_dif',
        )
        self.renderer.load_cubemap(
            'grey_cube_ref',
            cubemap_asset='grey_cube_ref',
        )
        self.renderer.load_image_light(
            'background',
            'grey_cube_dif',
            'grey_cube_ref',
            render_background=False,
        )
        self.renderer.set_active_image_light('background')
        '''
        #self.renderer.add_direction_light(
        #    'sun', (0,-1,0), (2,2,2))
        #self.renderer.set_ambient_color((0.5, 0.5, 0.5))
        
        self.renderer.set_ambient_color((1.,1.,1.))
        
        self.renderer.set_background_color(
            #(106./255., 223/255., 255./255.),
            (188./255., 225./255., 242./255.),
        )
    
    def mouse_button_callback(self, window, button, action, mods):
        if self._ctrl_down:
            if action == glfw.PRESS:
                mask = self.window.read_pixels()
                x, y = self.camera_control.get_mouse_pixel_position(window)
                fbw, fbh = self.window.framebuffer_size()
                y = fbh - y
                mask_color = mask[y,x]
                render_id = color_byte_to_index(mask_color) - 1
                if render_id in self._render_players:
                    player_id = self._render_players[render_id]
                    self.selected_player = player_id
                    self.print_player_info(player_id, self.report)
                else:
                    self.selected_player = None
                self._update_players()
                #self.window.set_active()
        else:
            return self.camera_control.mouse_callback(
                window, button, action, mods)
    
    def _init_callbacks(self):
        self._shift_down = False
        self._ctrl_down = False
        #self.window.set_mouse_button_callback(
        #    self.camera_control.mouse_callback)
        self.window.set_mouse_button_callback(
            self.mouse_button_callback)
        self.window.set_cursor_pos_callback(
            self.camera_control.mouse_move)
        self.window.set_scroll_callback(
            self.camera_control.scroll_callback)
        self.window.set_key_callback(self.key_callback)
    
    def step_to_block(self, step):
        s = (step-self.step_0)
        block_index = s // self.reports_per_block
        block_step = s % self.reports_per_block
        return block_index, block_step
    
    def change_step(self, step):
        step = max(self.step_0, min(self.step_N-1, step))
        block_index, block_step = self.step_to_block(step)
        if block_index != self.block_index:
            self.block_index = block_index
            report_block = load_example_data(
                self._example_report, self.report_files[self.block_index])
            self.current_report_block = self.get_report_block(report_block)
        self.current_step = step
        
        print(f'Current Step: {step} '
            f'Block Location: {block_index}, {block_step}')
        
        self.report = tree_getitem(
            self.current_report_block, block_step)
        
        #print(self.report)
        
        self._update_landscape()
        #self._update_water()
        if self.get_active_players is not None:
            self._update_players()
        
        if self.selected_player is not None:
            self.print_player_info(self.selected_player, self.report)
    
    def _update_players(self):
        #active_players = self.get_active_players(self.params, self.report)
        active_players = self.get_active_players(self.report)
        if not self.show_players:
            active_players = jnp.zeros_like(active_players)
        
        #player_x = self.get_player_x(self.params, self.report)
        #player_r = self.get_player_r(self.params, self.report)
        player_x = self.get_player_x(self.report)
        player_r = self.get_player_r(self.report)
        if self.get_player_energy is not None:
            #player_energy = self.get_player_energy(self.params, self.report)
            player_energy = self.get_player_energy(self.report)
        else:
            player_energy = np.zeros(player_r.shape)
        player_transforms = self._player_transform(player_x, player_r)
        
        print(f'Active Players: {jnp.sum(active_players)}')
        
        # figure out which players are in the frustum, and z sort them
        self._render_players = {i : -1 for i in range(self.max_players)}
        projection = self.renderer.get_projection()
        view_matrix = self.renderer.get_view_matrix()
        local_transforms = projection @ view_matrix @ player_transforms
        local_positions = local_transforms[:,:,3]
        screen_positions = local_positions[:,:2] / local_positions[:,[3]]
        in_bounds = (
            (jnp.abs(screen_positions[:,0]) <= 0.5) &
            (jnp.abs(screen_positions[:,1]) <= 0.5)
        )
        
        score = jnp.where(in_bounds, -local_positions[:,2], -jnp.inf)
        _, best_players = jax.lax.top_k(score, self.max_players)
        
        #for player_id in range(scene_players):
        #    if active_players[player_id]:
        #        pass
        next_render_id = 0
        scene_players, = active_players.shape
        for player_id in best_players: #range(scene_players):
            if active_players[player_id] & in_bounds[player_id]:
                self._render_players[next_render_id] = player_id
                next_render_id += 1
                if next_render_id not in self._render_players:
                    break
        
        #for player_id in range(self.max_players):
        #for render_id, player_id in render_players.items():
        for render_id, player_id in self._render_players.items():
            player_name = f'player_{render_id}'
            eye_white_name = f'player_eye_white_{render_id}'
            eye_pupil_name = f'player_eye_pupil_{render_id}'
            energy_background_name = f'player_energy_background_{render_id}'
            energy_name = f'player_energy_{render_id}'
            if player_id != -1 and active_players[player_id]:
                self.renderer.show_instance(player_name)
                self.renderer.set_instance_transform(
                    player_name, player_transforms[player_id])
                self.renderer.show_instance(eye_white_name)
                self.renderer.set_instance_transform(
                    eye_white_name, player_transforms[player_id])
                self.renderer.show_instance(eye_pupil_name)
                self.renderer.set_instance_transform(
                    eye_pupil_name, player_transforms[player_id])
                
                if (self.selected_player is not None and
                    player_id == self.selected_player
                ):
                    player_color = np.array([1., 0., 0.])
                else:
                    player_color = self.get_player_color(
                    #    player_id, self.params, self.report)
                        player_id, self.report)
                material_name = f'player_material_{render_id}'
                self.renderer.set_material_flat_color(
                    material_name, player_color)
                
                if self.get_player_energy is not None:
                    background_transform = player_transforms[player_id].copy()
                    background_transform[1,3] += PLAYER_RADIUS * 1.5
                    self.renderer.show_instance(energy_background_name)
                    self.renderer.set_instance_transform(
                        energy_background_name, background_transform)
                    
                    #energy_transform = background_transform.copy()
                    energy_scale = np.eye(4)
                    energy_scale[0,0] = player_energy[player_id]
                    energy_pivot = np.eye(4)
                    energy_pivot[0,3] = PLAYER_RADIUS-0.05
                    energy_anti_pivot = np.eye(4)
                    energy_anti_pivot[0,3] = -PLAYER_RADIUS+0.05
                    energy_transform = (
                        background_transform @
                        energy_pivot @
                        energy_scale @
                        energy_anti_pivot
                    )
                    self.renderer.show_instance(energy_name)
                    self.renderer.set_instance_transform(
                        energy_name, energy_transform)
                
            else:
                self.renderer.hide_instance(player_name)
                self.renderer.hide_instance(eye_white_name)
                self.renderer.hide_instance(eye_pupil_name)
                if self.get_player_energy is not None:
                    self.renderer.hide_instance(energy_background_name)
                    self.renderer.hide_instance(energy_name)
    
    def _update_landscape(self):
        self._update_sun_position()
        if self.get_terrain_texture is not None:
            texture = self.get_terrain_texture(
                #self.params,
                self.report,
                self.terrain_texture_resolution,
                self.display_mode,
            )
            self.renderer.load_texture(
                'terrain_texture',
                texture_data = texture,
            )
        
        if self.get_terrain_map:
            #self.terrain_map = self.get_terrain_map(self.params, self.report)
            self.terrain_map = self.get_terrain_map(self.report)
            (
                terrain_vertices,
                terrain_normals,
            ) = make_height_map_vertices_and_normals(
                self.terrain_map, spacing=self.mesh_spacing)
            self.renderer.load_mesh(
                name='terrain_mesh',
                mesh_data={
                    'vertices' : terrain_vertices,
                    'normals' : terrain_normals,
                    'faces' : self.terrain_faces,
                    'uvs' : self.terrain_uvs,
                }
            )
        
        if self.get_water_map:
            #water_map = self.get_water_map(self.params, self.report)
            water_map = self.get_water_map(self.report)
            self.total_height_map = self.terrain_map + water_map
            (
                water_vertices,
                water_normals,
            ) = make_height_map_vertices_and_normals(
                self.total_height_map, spacing=self.mesh_spacing)
            self.renderer.load_mesh(
                name='water_mesh',
                mesh_data={
                    'vertices' : water_vertices,
                    'normals' : water_normals,
                    'faces' : self.water_faces,
                    'uvs' : self.water_uvs,
                },
                color_mode='flat_color',
            )
        
        else:
            self.total_height_map = self.terrain_map
    
    def _player_transform(self, player_x, player_r):
        height, width = self.world_size
        zy = (player_x[..., 0] // self.mesh_spacing).astype(jnp.int32)
        zx = (player_x[..., 1] // self.mesh_spacing).astype(jnp.int32)
        z = self.total_height_map[zy, zx] + PLAYER_RADIUS
        y = player_x[..., 0] - height/2. + 0.5
        x = player_x[..., 1] - width/2. + 0.5
        
        cs = np.array((( 1, 0), ( 0, 1), (-1, 0), ( 0,-1)))[player_r]
        c = cs[...,0]
        s = cs[...,1]
        
        transforms = np.zeros((*player_r.shape, 4, 4))
        transforms[...,0,0] = c
        transforms[...,0,1] = s
        transforms[...,0,3] = x
        
        transforms[...,1,0] = -s
        
        transforms[...,1,1] = c
        transforms[...,1,3] = y
        
        transforms[...,2,2] = 1.
        transforms[...,2,3] = z
        
        transforms[...,3,3] = 1.
        
        return self.upright @ transforms
    
    def start(self):
        self.window.show_window()
        self.window.enable_window()
        
        while not self.window.should_close():
            self.window.poll_events()
            self.render()
            self.window.swap_buffers()
        
        glfw_context.terminate()
    
    def render(self):
        fbw, fbh = self.window.framebuffer_size()
        self.renderer.viewport_scissor(0,0,fbw,fbh)
        if self._ctrl_down:
            self.renderer.mask_render(flip_y=False)
        else:
            self.renderer.color_render(flip_y=False)
    
    def key_callback(self, window, key, scancode, action, mods):
        # 0-9 sets various display modes
        if action == glfw.PRESS and key >= 48 and key < 58:
            self.display_mode = (key - 48)
            print(f'display mode: {self.display_mode}')
            self.change_step(self.current_step)
        # -
        if action == glfw.PRESS and key == 45:
            self.step_size = max(1, self.step_size-1)
            print(f'step size: {self.step_size}')
        # +
        if action == glfw.PRESS and key == 61:
            self.step_size += 1
            print(f'step size: {self.step_size}')
        # shift
        if key in (340, 344):
            self._shift_down = action
        # ctrl
        if key in (341, 345):
            self._ctrl_down = action
        # a
        if key == 65 and action:
             self.show_players = not self.show_players
             self.change_step(self.current_step)
        
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == 44:
                if self._shift_down:
                    self.change_step(self.current_step - self.reports_per_block)
                else:
                    self.change_step(self.current_step - self.step_size)
            elif key == 46:
                if self._shift_down:
                    self.change_step(self.current_step + self.reports_per_block)
                else:
                    self.change_step(self.current_step + self.step_size)
        self.camera_control.key_callback(window, key, scancode, action, mods)
