import numpy as np

import jax.numpy as jnp
import jax.random as jrng

import glfw
import splendor.core as core
import splendor.contexts.glfw_context as glfw_context
from splendor.interactive_camera_glfw import InteractiveCameraGLFW
import splendor.camera as camera

#   o----o----o----o
#   |\   |\   |\   |
#   | \  | \  | \  |
#   |  \ |  \ |  \ |
#   |   \|   \|   \|
#   o----o----o----o
#   |\   |\   |\   |
#   | \  | \  | \  |
#   |  \ |  \ |  \ |
#   |   \|   \|   \|
#   o----o----o----o
#   |\   |\   |\   |
#   | \  | \  | \  |
#   |  \ |  \ |  \ |
#   |   \|   \|   \|
#   o----o----o----o

def make_height_map_mesh_old(height_map, slope_spacing=0.5):
    h,w = height_map.shape
    flat_spacing = 1. - slope_spacing
    half_flat_spacing = flat_spacing/2.
    
    # make vertices
    vertices = jnp.zeros((h,2,w,2,3))
    vertices = vertices.at[:,:,:,:,2].set(jnp.expand_dims(height_map, [1,3]))
    vertices = vertices.at[:,:,:,0,0].set(jnp.expand_dims(
        jnp.arange(-half_flat_spacing, w-half_flat_spacing), [0,1]))
    vertices = vertices.at[:,:,:,1,0].set(jnp.expand_dims(
        jnp.arange(half_flat_spacing, w+half_flat_spacing), [0,1]))
    vertices = vertices.at[:,0,:,:,1].set(jnp.expand_dims(
        jnp.arange(-half_flat_spacing, h-half_flat_spacing), [1,2]))
    vertices = vertices.at[:,1,:,:,1].set(jnp.expand_dims(
        jnp.arange(half_flat_spacing, h+half_flat_spacing), [1,2]))
    
    # make uvs
    uvs = jnp.zeros((h*2, w*2, 2))
    uvs = uvs.at[:,:,0].set(jnp.linspace(0,1,num=h*2)[:,None])
    uvs = uvs.at[:,:,1].set(jnp.linspace(0,1,num=w*2)[None,:])
    
    # make normals
    padded_height_map = jnp.pad(height_map, pad_width=1, mode='edge')
    y_gradient = -(padded_height_map[2:,:] - padded_height_map[:-2,:])[:,1:-1]
    x_gradient = -(padded_height_map[:,2:] - padded_height_map[:,:-2])[1:-1,:]
    z_gradient = (1. - x_gradient**2 - y_gradient**2)**0.5
    normals = jnp.stack((y_gradient, x_gradient, z_gradient), axis=-1)
    
    # make faces
    faces = jnp.zeros((h*2-1,w*2-1,4), dtype=jnp.int32)
    faces = faces.at[:,:,0].add(jnp.expand_dims(
        jnp.arange(0, w*2-1), [0]))
    faces = faces.at[:,:,1].add(jnp.expand_dims(
        jnp.arange(1, w*2), [0]))
    faces = faces.at[:,:,2].add(jnp.expand_dims(
        jnp.arange(1, w*2) + w*2, [0]))
    faces = faces.at[:,:,3].add(jnp.expand_dims(
        jnp.arange(0, w*2-1) + w*2, [0]))
    
    faces = faces.at[:,:,:].add(jnp.expand_dims(
        jnp.arange(0, h*2-1) * (w*2), [1,2]))
    
    return (
        vertices.reshape(-1,3),
        uvs.reshape(-1,2),
        normals.reshape(-1, 3),
        faces.reshape(-1,4),
    )

def make_height_map_mesh(height_map):
    h, w = height_map.shape
    
    vertices = jnp.zeros((h, w, 3), dtype=jnp.float32)
    vertices = vertices.at[:,:,0].set(jnp.arange(w)[None,:] - w/2.)
    vertices = vertices.at[:,:,1].set(jnp.arange(h)[:,None] - h/2.)
    vertices = vertices.at[:,:,2].set(height_map)
    vertices = vertices.reshape(-1,3)
    
    normals = jnp.zeros((h, w, 3), dtype=jnp.float32)
    # va (2, 0, dx)
    lo_x_i = jnp.clip(jnp.arange(w)-1, min=0)
    lo_x = height_map[:,lo_x_i]
    hi_x_i = jnp.arange(w)+1
    hi_x = height_map[:,hi_x_i]
    dx = (hi_x - lo_x)/2.
    # vb (0, 2, dy)
    lo_y_i = jnp.clip(jnp.arange(h)-1, min=0)
    lo_y = height_map[lo_y_i]
    hi_y_i = jnp.arange(h)+1
    hi_y = height_map[hi_y_i]
    dy = (hi_y - lo_y)/2.
    # cross product (ya*zb - za*yb), (za*xb - xa*zb), (xa*yb - ya*xb)
    normals = normals.at[:,:,0].set(-dx)
    normals = normals.at[:,:,1].set(-dy)
    normals = normals.at[:,:,2].set(1)
    normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals.reshape(-1,3)
    
    uvs = jnp.zeros((h, w, 2), dtype=jnp.float32)
    uvs = uvs.at[:,:,0].set(jnp.arange(w)[None, :])
    uvs = uvs.at[:,:,1].set(jnp.arange(h)[:,None])
    uvs = uvs.reshape(-1,2)
    
    faces = jnp.zeros((2, h-1, w-1, 3), dtype=jnp.int32)
    faces = faces.at[0,:,:,0].add(jnp.arange(0, w-1)[None, :])
    faces = faces.at[0,:,:,1].add(jnp.arange(1, w)[None, :])
    faces = faces.at[0,:,:,2].add(jnp.arange(1, w)[None, :] + w)
    
    faces = faces.at[1,:,:,0].add(jnp.arange(0, w-1)[None, :])
    faces = faces.at[1,:,:,1].add(jnp.arange(1, w)[None, :] + w)
    faces = faces.at[1,:,:,2].add(jnp.arange(0, w-1)[None, :] + w)
    
    faces = faces.at[:,:,:,:].add(jnp.arange(0, h-1)[:,None,None] * w)
    faces = faces.reshape(-1,3)
    
    return vertices, normals, uvs, faces

def make_obj(vertices, normals, uvs, faces, file_path=None):
    lines = []
    for x, y, z in vertices:
        lines.append(f'v {float(x)} {float(y)} {float(z)}')
    
    for u, v in vertices:
        lines.append(f'vt {float(u)} {float(v)}')
    
    for x, y, z in normals:
        lines.append(f'vn {float(x)} {float(y)} {float(z)}')
    
    for face_vertices in faces:
        lines.append('f ' + ' '.join(
            [f'{int(v+1)}/{int(v+1)}/{int(v+1)}' for v in face_vertices]
        ))
    
    text = '\n'.join(lines)
    
    if file_path is not None:
        with open(file_path, 'w') as f:
            f.write(text)
    
    return text

def start_terrain_viewer(
    terrain_maps,
    water_maps,
    water_offset=-0.01,
    window_width=512,
    window_height=512,
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
    c = np.cos(np.radians(-45))
    s = np.sin(np.radians(-45))
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, c,-s, h],
        [0, s, c, h],
        [0, 0, 0, 1],
    ])
    renderer.set_view_matrix(np.linalg.inv(camera_pose))
    
    camera_control = InteractiveCameraGLFW(window, renderer)
    for i, (t, w) in enumerate(zip(terrain_maps, water_maps)):
        w = t + w - water_offset
        tv, tn, tuv, tf = make_height_map_mesh(t)
        wv, wn, wuv, wf = make_height_map_mesh(w)
        wv = wv.at[:,2].add(water_offset)
        
        #tv = tv * 0.1
        #wv = wv * 0.1
        
        renderer.load_mesh(
            name=f'terrain_mesh_{i}',
            mesh_data={
                'vertices':np.array(tv),
                'normals':np.array(tn),
                'uvs':np.array(tuv),
                'faces':np.array(tf),
            },
            color_mode='flat_color',
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
    
    viewer_state = {
        'renderer' : renderer,
        'current_frame' : 0,
    }
    
    def render():
        current_frame = viewer_state['current_frame']
        renderer = viewer_state['renderer']
        
        renderer.set_instance_mesh(
            'terrain_instance', f'terrain_mesh_{current_frame}')
        renderer.set_instance_mesh(
            'water_instance', f'water_mesh_{current_frame}')
        
        fbw, fbh = window.framebuffer_size()
        renderer.viewport_scissor(0,0,fbw,fbh)
        renderer.color_render(flip_y=False)
    
    def key_callback(window, key, scancode, action, mods):
        current_frame = viewer_state['current_frame']
        if action == glfw.PRESS:
            if key == 44: # ,
                viewer_state['current_frame'] = (current_frame - 1) % n
            elif key == 46: # .
                viewer_state['current_frame'] = (current_frame + 1) % n
        
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

def visualize_water_flow(key):
    from geology import fractal_noise
    from water import flow_step, flow_step_twodir
    
    world_size=(128,512)
    
    terrain = fractal_noise(
        key=key,
        world_size=world_size,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        grid_unit_scale=0.005,
        height_scale=50,
    )
    water = jnp.full(world_size, 0.5)
    flow_rate = 0.25
    
    terrain_maps = [terrain]
    water_maps = [water]
    for i in range(2048):
        water = flow_step_twodir(terrain, water, flow_rate)
        if i % 64 == 0:
            terrain_maps.append(terrain)
            water_maps.append(water)

    start_terrain_viewer(
        terrain_maps,
        water_maps,
        window_height=1024,
        window_width=1024,
    )

def visualize_erosion(key):
    from geology import fractal_noise
    from erosion import simulate_erosion
    
    world_size=(1024,1024)
    
    terrain = fractal_noise(
        key=key,
        world_size=world_size,
        octaves=12,
        persistence=1./1.5, #0.5,
        lacunarity=1.5, #2.0,
        grid_unit_scale=0.005,
        height_scale=50,
    )
    water_initial = 0.1
    water_flow_rate = 0.25
    erosion_steps = 100
    erosion_endurance = 0.0 #0.01 #0.2
    erosion_ratio = 0.25 #0.25/erosion_steps
    
    water = jnp.full(world_size, water_initial)
    
    macro_steps = 32
    visualize_freq = 1
    rain_freq = 8
    rain_percent = 0.05
    terrain_maps = [terrain]
    water_maps = [water]
    for i in range(macro_steps):
        print('macro step', i, '/', macro_steps)
        
        if i % rain_freq == 0:
            total_water = jnp.sum(water)
            total_locations = world_size[0] * world_size[1]
            water = water * (1.-rain_percent) + rain_percent * total_water / total_locations
        
        erosion_initial = jnp.zeros(world_size)
        terrain, water = simulate_erosion(
            terrain,
            water,
            erosion_initial,
            water_flow_rate,
            erosion_steps,
            erosion_endurance,
            erosion_ratio,
        )
        
        if i % visualize_freq == 0:
            terrain_maps.append(terrain)
            water_maps.append(water)
    
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
    #visualize_water_flow(key)
    visualize_erosion(key)
