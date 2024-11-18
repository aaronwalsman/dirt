import jax.numpy as jnp

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
    
    '''
    triangles = jnp.zeros((h*2-1,w*2-1,2,3), dtype=jnp.int32)
    
    triangles = triangles.at[:,:,0,0].add(jnp.expand_dims(
        jnp.arange(0, w*2-1), [0]))
    triangles = triangles.at[:,:,0,1].add(jnp.expand_dims(
        jnp.arange(1, w*2), [0]))
    triangles = triangles.at[:,:,0,2].add(jnp.expand_dims(
        jnp.arange(1, w*2) + w*2, [0]))
    
    triangles = triangles.at[:,:,1,0].add(jnp.expand_dims(
        jnp.arange(0, w*2-1), [0]))
    triangles = triangles.at[:,:,1,1].add(jnp.expand_dims(
        jnp.arange(1, w*2) + w*2, [0]))
    triangles = triangles.at[:,:,1,2].add(jnp.expand_dims(
        jnp.arange(0, w*2-1) + w*2, [0]))
    
    triangles = triangles.at[:,:,0,:].add(jnp.expand_dims(
        jnp.arange(0, h*2-1) * (w*2), [1,2]))
    #triangles = triangles.at[:,:,0,1].add(jnp.expand_dims(
    #    jnp.arange(1, h*2) * (w*2), [1]))
    #triangles = triangles.at[:,:,0,2].add(jnp.expand_dims(
    #    (jnp.arange(1, h*2) + h*2) * (w*2), [1]))
    
    triangles = triangles.at[:,:,1,:].add(jnp.expand_dims(
        jnp.arange(0, h*2-1) * (w*2), [1,2]))
    #triangles = triangles.at[:,:,1,1].add(jnp.expand_dims(
    #    (jnp.arange(1, h*2) + h*2) * (w*2), [1]))
    #triangles = triangles.at[:,:,1,2].add(jnp.expand_dims(
    #    (jnp.arange(0, h*2-1) + h*2) * (w*2), [1]))
    '''
    
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
    
    return vertices.reshape(-1,3), faces.reshape(-1,4)

def make_obj(vertices, faces, file_path=None):
    lines = []
    for x, y, z in vertices:
        lines.append(f'v {float(x)} {float(y)} {float(z)}')
    
    for face_vertices in faces:
        lines.append('f ' + ' '.join([f'{int(v+1)}' for v in face_vertices]))
    
    text = '\n'.join(lines)
    
    if file_path is not None:
        with open(file_path, 'w') as f:
            f.write(text)
    
    return text

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
