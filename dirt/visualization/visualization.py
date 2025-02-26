import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

import glfw
import splendor.core as core
import splendor.contexts.glfw_context as glfw_context
from splendor.interactive_camera_glfw import InteractiveCameraGLFW
import splendor.camera as camera
from splendor.image import save_image

import dirt.gridworld2d.spawn as spawn

def add_players(
    renderer,
    color_indices,
    color_index_values,
):
    renderer.load_mesh(
        name='player_mesh',
        mesh_primitive={
            'shape':'sphere',
            'radius':1,
        },
        color_mode='flat_color',
    )
    
    for i, color_index_value in enumerate(color_index_values):
        material_name = f'player_material_{i}'
        renderer.load_material(material_name, flat_color=color_index_value)
    
    player_instances = []
    for i, color_index in enumerate(color_indices):
        transform = np.eye(4)
        renderer.add_instance(
            f'player_{i}',
            mesh_name='player_mesh',
            material_name=f'player_material_{color_index}',
            transform=transform
        )
        player_instances.append(f'player_{i}')
    
    return player_instances

def update_player_locations(renderer, locations, height_map):
    rotate_upright = np.array([
        [ 1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0,-1, 0, 0],
        [ 0, 0, 0, 1],
    ])
    world_size = height_map.shape
    half_h, half_w = world_size[0]//2, world_size[1]//2
    for i, location in enumerate(locations):
        transform = np.eye(4)
        transform[0,3] = location[1] - half_w
        transform[1,3] = location[0] - half_h
        transform[2,3] = height_map[location[0], location[1]]
        transform = rotate_upright @ transform
        renderer.set_instance_transform(f'player_{i}', transform)

def start_terrain_viewer(
    terrain_maps,
    water_maps,
    water_offset=-0.01,
    window_width=512,
    window_height=512,
    camera_distance=None,
    player_locations=[],
    player_color_indices=[],
    player_color_values=[],
    texture_terrain=False,
):
    glfw_context.initialize()
    window = glfw_context.GLFWWindowWrapper(
        width=window_width,
        height=window_height,
        anti_alias=False,
        anti_alias_samples=0,
    )
    renderer = core.SplendorRender()
    window.show_window()
    window.enable_window()
    
    n = len(terrain_maps)
    
    projection = camera.projection_matrix(
        np.radians(90.), 1., near_clip=1., far_clip=5000)
    renderer.set_projection(projection)
    
    h,w = terrain_maps[0].shape
    if camera_distance is None:
        camera_distance = h
    c = np.cos(np.radians(-45))
    s = np.sin(np.radians(-45))
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, c,-s, camera_distance],
        [0, s, c, camera_distance],
        [0, 0, 0, 1],
    ])
    renderer.set_view_matrix(np.linalg.inv(camera_pose))
    
    camera_control = InteractiveCameraGLFW(window, renderer)
    for i, (t, w) in enumerate(zip(terrain_maps, water_maps)):
        w = t + w - water_offset
        tv, tn, tuv, tf = make_height_map_mesh(t)
        wv, wn, wuv, wf = make_height_map_mesh(w)
        wv = wv.at[:,2].add(water_offset)
        
        if texture_terrain:
            color_mode='textured'
        else:
            color_mode='flat_color'
        
        renderer.load_mesh(
            name=f'terrain_mesh_{i}',
            mesh_data={
                'vertices':np.array(tv),
                'normals':np.array(tn),
                'uvs':np.array(tuv),
                'faces':np.array(tf),
            },
            color_mode=color_mode,
        )
        renderer.load_mesh(
            name=f'water_mesh_{i}',
            mesh_data={
                'vertices':np.array(wv),
                'normals':np.array(wn),
                'uvs':np.array(wuv),
                'faces':np.array(wf),
            },
            color_mode='flat_color',
        )
    
    if texture_terrain:
        texture = height_texture(terrain_maps[0], [1,0.25,0.25], [1,1,1])
        renderer.load_texture(
            name='terrain_texture',
            texture_data = texture,
        )
        renderer.load_material(
            name='terrain_material',
            texture_name='terrain_texture',
        )
    else:
        renderer.load_material(
            name='terrain_material',
            flat_color=(0.5,0.5,0.5),
        )
    
    renderer.load_material(
        name='water_material',
        flat_color=(0.25,0.25,0.75),
    )
    
    rotate_upright = np.array([
        [ 1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0,-1, 0, 0],
        [ 0, 0, 0, 1],
    ])
    renderer.add_instance(
        'terrain_instance',
        mesh_name='terrain_0',
        material_name='terrain_material',
        transform=rotate_upright,
    )
    renderer.add_instance(
        'water_instance',
        mesh_name='water_0',
        material_name='water_material',
        transform=rotate_upright,
    )
    
    renderer.load_cubemap(
        'grey_cube_dif',
        cubemap_asset='grey_cube_dif',
    )
    renderer.load_cubemap(
        'grey_cube_ref',
        cubemap_asset='grey_cube_ref',
    )
    renderer.load_image_light(
        'background',
        'grey_cube_dif',
        'grey_cube_ref',
        render_background=False,
    )
    renderer.set_active_image_light('background')
    
    player_instances = add_players(
        renderer, player_color_indices, player_color_values)
    update_player_locations(
        renderer,
        player_locations,
        np.array(terrain_maps[0]+water_maps[0])
    )
    
    viewer_state = {
        'renderer' : renderer,
        'current_frame' : 0,
        'view_water' : True,
    }
    
    def render():
        current_frame = viewer_state['current_frame']
        renderer = viewer_state['renderer']
        
        renderer.set_instance_mesh(
            'terrain_instance', f'terrain_mesh_{current_frame}')
        renderer.set_instance_mesh(
            'water_instance', f'water_mesh_{current_frame}')
        
        if viewer_state['view_water']:
            instances = ['terrain_instance', 'water_instance']
        else:
            instances = ['terrain_instance']
        
        instances.extend(player_instances)
        
        fbw, fbh = window.framebuffer_size()
        renderer.viewport_scissor(0,0,fbw,fbh)
        renderer.color_render(flip_y=False, instances=instances)
    
    def key_callback(window, key, scancode, action, mods):
        current_frame = viewer_state['current_frame']
        if action == glfw.PRESS:
            if key == 44: # ,
                viewer_state['current_frame'] = (current_frame - 1) % n
            elif key == 46: # .
                viewer_state['current_frame'] = (current_frame + 1) % n
            elif key == 87: # w
                viewer_state['view_water'] = not viewer_state['view_water']
            
            elif key == 83: #s
                color = camera_control.window.read_pixels()
                save_image(color[::-1,:,:3], 'image_%04i.png'%current_frame)
            
            if key == 44 or key == 46:
                height_map = np.array(
                    terrain_maps[viewer_state['current_frame']] +
                    water_maps[viewer_state['current_frame']]
                )
                update_player_locations(
                    renderer,
                    player_locations,
                    height_map,
                )
        
        camera_control.key_callback(window, key, scancode, action, mods)
    
    window.set_mouse_button_callback(camera_control.mouse_callback)
    window.set_cursor_pos_callback(camera_control.mouse_move)
    window.set_key_callback(key_callback)
    window.set_scroll_callback(camera_control.scroll_callback)
    
    while not window.should_close():
        window.poll_events()
        render()
        window.swap_buffers()
    
    glfw_context.terminate()

def height_texture(height_map, low_color, high_color):
    low_color = jnp.array(low_color).reshape(3)
    high_color = jnp.array(high_color).reshape(3)
    height_min = jnp.min(height_map)
    height_max = jnp.max(height_map)
    t = (height_map - height_min) / (height_max - height_min)
    t = t[:,:,None]
    texture = ((1-t) * low_color + t * high_color) * 255
    texture = np.array(texture.astype(jnp.uint8))
    return texture

def visualize_water_flow(key):
    from geology import fractal_noise
    from water import flow_step, flow_step_twodir
    
    # generate terrain
    world_size=(512,512)
    terrain = fractal_noise(
        key=key,
        world_size=world_size,
        octaves=12, #12
        persistence=0.5,
        lacunarity=2.0,
        grid_unit_scale=0.0025,
        height_scale=100,
    )
    
    # initailize the water
    initial_water = 2.
    flow_rate = 0.25
    water = jnp.full(world_size, initial_water)
    
    # generate the water steps
    visualization_steps = 64 
    steps_per_visualization = 256
    def generate_maps(water, i):
        jax.debug.print('computing {i}/{steps}', i=i, steps=visualization_steps)
        def step(water, _):
            water = flow_step_twodir(terrain, water, flow_rate)
            return water, None
        
        water, _ = jax.lax.scan(
            step, water, None, length=steps_per_visualization)
        
        return water, water
    
    terrain_maps = [terrain] * visualization_steps
    water, water_maps = jax.lax.scan(
        generate_maps, water, jnp.arange(visualization_steps))
    
    player_density = 0.0
    num_players = int((world_size[0] * world_size[1]) * player_density)
    player_color_indices = np.zeros(num_players, dtype=int)
    player_color_values = [[0,0.8,0]]
    key, spawn_key = jrng.split(key)
    player_locations = np.array(spawn.unique_x(
        spawn_key, num_players, world_size))
    
    # start the viewer
    start_terrain_viewer(
        terrain_maps,
        water_maps,
        window_height=1024,
        window_width=1024,
        camera_distance=1024,
        player_locations=player_locations,
        player_color_indices=player_color_indices,
        player_color_values=player_color_values,
        texture_terrain=True,
    )

def visualize_erosion(key):
    from geology import fractal_noise
    from water import flow_step_twodir
    from erosion import simulate_erosion
    
    # generate the initial terrain
    world_size=(1024,1024)
    terrain = fractal_noise(
        key=key,
        world_size=world_size,
        octaves=12,
        persistence=0.5,
        lacunarity=2.0,
        grid_unit_scale=0.0025,
        height_scale=100,
    )
    initial_terrain = terrain
    
    # initialize the water
    evaporation = 0.01
    initial_water = 2.0
    water_flow_rate = 0.1 #0.25
    water = jnp.full(world_size, initial_water)
    
    # step the water to form the initial lakes
    #initial_water_steps = 1024 * 64
    #def initialize_water(water, _):
    #    water = flow_step_twodir(terrain, water, water_flow_rate)
    #    return water, None
    #water, _ = jax.lax.scan(
    #    initialize_water, water, None, length=initial_water_steps)
    
    # initialize the water already all the way downhill
    print('initializing water')
    total_water = initial_water * world_size[0] * world_size[1]
    water_level = jnp.min(terrain)
    water = jnp.zeros_like(terrain)
    while jnp.sum(water) < total_water:
        water_level += 0.001
        water = water_level - terrain
        water = jnp.where(water < 0., 0, water)
    initial_water = water
    print('finished initializing water')
    
    # erosion parameters
    erosion_steps = 1024
    erosion_endurance = 0. #0.01 #0.2
    erosion_ratio = 0.1 #0.25/erosion_steps
    accumulated_erosion = jnp.zeros_like(water)
    
    # generate the water and terrain steps
    visualization_steps = 64
    
    def generate_maps(state, i):
        jax.debug.print('computing {i}/{steps}', i=i, steps=visualization_steps)
        
        # evaporate and rain a little bit
        # (this will get replaced by the weather next)
        terrain, water, accumulated_erosion = state
        total_water = jnp.sum(water)
        total_locations = world_size[0] * world_size[1]
        water = (
            water * (1. - evaporation) +
            (total_water / total_locations) * evaporation
        )
        
        # simulate the erosion
        terrain, water, accumulated_erosion = simulate_erosion(
            terrain,
            water,
            accumulated_erosion,
            water_flow_rate,
            erosion_steps,
            erosion_endurance,
            erosion_ratio,
        )
        
        return (terrain, water, accumulated_erosion), (terrain, water)
    
    state, terrain_water_maps = jax.lax.scan(
        generate_maps,
        (terrain, water, accumulated_erosion),
        jnp.arange(visualization_steps),
    )
    
    terrain_maps, water_maps = terrain_water_maps
    
    terrain_maps = jnp.concatenate((initial_terrain[None,...], terrain_maps))
    water_maps = jnp.concatenate((initial_water[None,...], water_maps))
    
    start_terrain_viewer(
        terrain_maps,
        water_maps,
        window_width=1024,
        window_height=1024,
    )

if __name__ == '__main__':
    #z = jnp.zeros((2,2))
    #z = z.at[0,0].set(0.)
    #z = z.at[0,1].set(0.5)
    #z = z.at[1,0].set(1.)
    #z = z.at[1,1].set(0.25)
    
    '''
    z = jnp.zeros((256, 256))
    z = z.at[:,:].add(jnp.expand_dims(jnp.sin(jnp.arange(0,16,1/16.)), [0]))
    z = z.at[:,:].add(jnp.expand_dims(jnp.sin(jnp.arange(0,16,1/16.)), [1]))
    z = z * 5
    vertices, faces = make_height_map_mesh(z)
    make_obj(vertices, faces, './tmp.obj')
    '''
    
    key = jrng.key(1022)
    visualize_water_flow(key)
    #visualize_erosion(key)
