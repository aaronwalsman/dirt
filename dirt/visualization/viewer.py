import numpy as np

import jax.numpy as jnp

import glfw
import splendor.core as core
import splendor.contexts.glfw_context as glfw_context
from splendor.interactive_camera_glfw import InteractiveCameraGLFW
import splendor.camera as camera
from splendor.masks import color_index_to_float
import splendor.primitives as primitives
from splendor.image import save_image

from mechagogue.tree import tree_len, tree_getitem
from mechagogue.serial import load_example_data
from mechagogue.arg_wrappers import ignore_unused_args

from dirt.visualization.height_map import make_height_map_mesh

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
default_get_water_map = lambda : None

PLAYER_RADIUS=0.4

class Viewer:
    def __init__(
        self,
        example_params,
        params_file,
        example_report,
        report_files,
        window_width=512,
        window_height=512,
        step_0 = 0,
        start_step=0,
        terrain_texture_size=(512,512),
        get_active_players=default_get_active_players,
        get_player_x=default_get_player_x,
        get_player_r=default_get_player_r,
        get_player_energy=None,
        get_terrain_map=default_get_terrain_map,
        get_terrain_texture=None,
        get_water_map=default_get_water_map,
    ):
        
        self.get_active_players = ignore_unused_args(
            get_active_players, ('params', 'report'))
        self.get_player_x = ignore_unused_args(
            get_player_x, ('params', 'report'))
        self.get_player_r = ignore_unused_args(
            get_player_r, ('params', 'report'))
        self.get_player_energy = get_player_energy
        if get_player_energy:
            self.get_player_energy = ignore_unused_args(
                self.get_player_energy, ('params', 'report'))
        self.get_terrain_map = ignore_unused_args(
            get_terrain_map, ('params', 'report'))
        self.get_terrain_texture = get_terrain_texture
        if self.get_terrain_texture:
            self.get_terrain_texture = ignore_unused_args(
                self.get_terrain_texture, ('params', 'report', 'texture_size'))
        self.get_water_map = ignore_unused_args(
            get_water_map, ('params', 'report'))
        
        self._init_params_and_reports(
            example_params,
            params_file,
            example_report,
            report_files,
            step_0=step_0,
            start_step=start_step,
        )
        self._init_context_and_window(window_width, window_height)
        self._init_splendor_render()
        self._init_terrain(terrain_texture_size)
        self._init_players()
        self._init_camera_and_lights()
        self._init_callbacks()
        
        self.step_size = 1
        self.change_step(start_step)
    
    def _init_params_and_reports(
        self,
        example_params,
        params_file,
        example_report,
        report_files,
        step_0,
        start_step,
    ):
        self.step_0 = step_0
        self.current_step = start_step
        self.block_index = 0
        
        self.params = load_example_data(example_params, params_file)
        self.report_files = report_files
        self.current_report_block = load_example_data(
            example_report, self.report_files[self.block_index])
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
    
    def _init_splendor_render(self):
        self.renderer = core.SplendorRender()
        self.upright = np.array([
            [ 1, 0, 0, 0],
            [ 0, 0, 1, 0],
            [ 0,-1, 0, 0],
            [ 0, 0, 0, 1],
        ])
    
    def _init_terrain(self, texture_size):
        self.terrain_texture_size=texture_size
        self.terrain_map = self.get_terrain_map(self.params, self.report)
        h, w = self.terrain_map.shape
        self.world_size = (h,w)
        
        vertices, normals, uvs, faces = make_height_map_mesh(self.terrain_map)
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
            texture_data=np.full((texture_size + (3,)), 127, dtype=np.uint8),
        )
        
        self.renderer.load_material(
            name='terrain_material',
            #flat_color=(0.5,0.5,0.5),
            texture_name='terrain_texture',
        )
        
        self.renderer.add_instance(
            name='terrain',
            mesh_name='terrain_mesh',
            material_name='terrain_material',
            transform=self.upright,
        )
    
    def _init_players(self):
        active_players = self.get_active_players(self.params, self.report)
        self.max_players, = active_players.shape
        
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
            player_color = color_index_to_float(player_id+1)
            self.renderer.load_material(
                name=material_name,
                flat_color=player_color,
            )
        
            player_name = f'player_{player_id}'
            self.renderer.add_instance(
                name=player_name,
                mesh_name='player_mesh',
                material_name=material_name,
                transform=np.eye(4),
                hidden=True,
            )
            
            player_eye_white_name = f'player_eye_white_{player_id}'
            self.renderer.add_instance(
                name=player_eye_white_name,
                mesh_name='eye_white_mesh',
                material_name='eye_white_material',
                transform=np.eye(4),
                hidden=True,
            )
            
            player_eye_pupil_name = f'player_eye_pupil_{player_id}'
            self.renderer.add_instance(
                name=player_eye_pupil_name,
                mesh_name='eye_pupil_mesh',
                material_name='eye_pupil_material',
                transform=np.eye(4),
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
            [1, 0, 0, 0],
            [0, c,-s, d],
            [0, s, c, d],
            [0, 0, 0, 1],
        ])
        
        self.renderer.set_view_matrix(np.linalg.inv(camera_pose))
        
        self.camera_control = InteractiveCameraGLFW(self.window, self.renderer)
        
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
        
        self.renderer.set_ambient_color((0.2, 0.2, 0.2))
        
    def _init_callbacks(self):
        self._shift_down = False
        self.window.set_mouse_button_callback(
            self.camera_control.mouse_callback)
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
            self.current_report_block = load_example_data(
                self.current_report_block, self.report_files[self.block_index])
        self.current_step = step
        
        print(f'Current Step: {step} '
            f'Block Location: {block_index}, {block_step}')
        
        self.report = tree_getitem(
            self.current_report_block, block_step)
        
        #print(self.report)
        
        self._update_terrain()
        #self._update_water()
        self._update_players()
    
    def _update_players(self):
        active_players = self.get_active_players(self.params, self.report)
        player_x = self.get_player_x(self.params, self.report)
        player_r = self.get_player_r(self.params, self.report)
        player_energy = self.get_player_energy(self.params, self.report)
        player_transforms = self._player_transform(player_x, player_r)
        
        print(f'Active Players: {jnp.sum(active_players)}')
        
        for player_id in range(self.max_players):
            player_name = f'player_{player_id}'
            eye_white_name = f'player_eye_white_{player_id}'
            eye_pupil_name = f'player_eye_pupil_{player_id}'
            energy_background_name = f'player_energy_background_{player_id}'
            energy_name = f'player_energy_{player_id}'
            if active_players[player_id]:
                self.renderer.show_instance(player_name)
                self.renderer.set_instance_transform(
                    player_name, player_transforms[player_id])
                self.renderer.show_instance(eye_white_name)
                self.renderer.set_instance_transform(
                    eye_white_name, player_transforms[player_id])
                self.renderer.show_instance(eye_pupil_name)
                self.renderer.set_instance_transform(
                    eye_pupil_name, player_transforms[player_id])
                
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
    
    def _update_terrain(self):
        if self.get_terrain_texture is not None:
            texture = self.get_terrain_texture(
                self.params, self.report, self.terrain_texture_size)
            self.renderer.load_texture(
                'terrain_texture',
                texture_data = texture,
            )
    
    def _player_transform(self, player_x, player_r):
        height, width = self.world_size
        y = player_x[..., 0]
        x = player_x[..., 1]
        z = self.terrain_map[y, x] + PLAYER_RADIUS
        y = y - height/2.
        x = x - width/2.
        
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
    
    def begin(self):
        self.window.show_window()
        self.window.enable_window()
        
        while not self.window.should_close():
            self.window.poll_events()
            self.render()
            self.window.swap_buffers()
        
        glfw_context.terminate()
    
    def render(self):
        #instances = ['terrain', 'tmp_sphere']
        fbw, fbh = self.window.framebuffer_size()
        self.renderer.viewport_scissor(0,0,fbw,fbh)
        self.renderer.color_render(flip_y=False)
    
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS and key == 45:
            self.step_size = max(1, self.step_size-1)
            print(f'step size: {self.step_size}')
        if action == glfw.PRESS and key == 61:
            self.step_size += 1
            print(f'step size: {self.step_size}')
        if key in (340, 344):
            self._shift_down = action
        
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
