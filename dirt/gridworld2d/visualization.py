import numpy as np

import jax.numpy as jnp

import splendor.contexts.egl as egl
import splendor.core as core
from splendor.image import save_image
from splendor.frame_buffer import FrameBufferWrapper

'''
o----o----o----o
|\   |\   |\   |
| \  | \  | \  |
|  \ |  \ |  \ |
|   \|   \|   \|
o----o----o----o
|\   |\   |\   |
| \  | \  | \  |
|  \ |  \ |  \ |
|   \|   \|   \|
o----o----o----o
|\   |\   |\   |
| \  | \  | \  |
|  \ |  \ |  \ |
|   \|   \|   \|
o----o----o----o
'''

def make_height_map_mesh(height_map, slope_spacing=0.5):
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

def make_obj(vertices, normals, faces, file_path=None):
    lines = []
    for x, y, z in vertices:
        lines.append(f'v {float(x)} {float(y)} {float(z)}')
    
    for x, y, z in normals:
        lines.append(f'vn {float(x)} {float(y)} {float(z)}')
    
    for face_vertices in faces:
        lines.append(
            'f ' + ' '.join([f'{int(v+1)}//{int(v+1)}' for v in face_vertices]))
    
    text = '\n'.join(lines)
    
    if file_path is not None:
        with open(file_path, 'w') as f:
            f.write(text)
    
    return text

def visualize_height_water(terrain, water):
    tv, tu, tn, tf = make_height_map_mesh(terrain)
    wv, wu, wn, wf = make_height_map_mesh(terrain + water)
    
    wv.at[...,2].add(-0.01)
    
    egl.initialize_plugin()
    egl.initialize_device(None)
    
    framebuffer = FrameBufferWrapper(
        512, 512,
        anti_alias=True,
        anti_alias_samples=8,
    )
    framebuffer.enable()
    renderer = core.SplendorRender()
    
    renderer.load_mesh(
        'terrain_mesh',
        mesh_data={
            'vertices':tv,
            'uvs':tu,
            'normals':tn,
            'faces':tf,
        },
        color_mode='flat_color',
    )
    
    renderer.load_material(
        'terrain_mat',
        flat_color=(0.5, 0.5, 0.5),
    )
    
    renderer.add_instance(
        'terrain', 'terrain_mesh', 'terrain_mat')
    
    renderer.add_direction_light('light', (0,0,-1), (1,1,1))
    
    camera_transform = np.eye(4)
    camera_transform[2,3] = 100
    camera_projection = camera.projection_matrix(
        np.radians(90), 1., far_clip=1000)
    
    renderer.set_view_matrix(np.linalg.inv(camera_transform))
    renderer.set_projection(camera_projection)
    
    renderer.color_render()
    
    image = frame_buffer.read_pixels()
    
    save_image(image, 'terrain.png')

if __name__ == '__main__':
    #z = jnp.zeros((2,2))
    #z = z.at[0,0].set(0.)
    #z = z.at[0,1].set(0.5)
    #z = z.at[1,0].set(1.)
    #z = z.at[1,1].set(0.25)
    
    z = jnp.zeros((256, 256))
    z = z.at[:,:].add(jnp.expand_dims(jnp.sin(jnp.arange(0,16,1/16.)), [0]))
    z = z.at[:,:].add(jnp.expand_dims(jnp.sin(jnp.arange(0,16,1/16.)), [1]))
    z = z * 5
    vertices, faces = make_height_map_mesh(z)
    make_obj(vertices, faces, './tmp.obj')
