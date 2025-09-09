import jax.numpy as jnp

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

def make_height_map_vertices_and_normals(height_map, spacing=1):
    h, w = height_map.shape
    bh = h + 2
    bw = w + 2
    
    vertices = jnp.zeros((bh, bw, 3), dtype=jnp.float32)
    vertices = vertices.at[:,:,0].set(
        spacing * (jnp.arange(bw)[None,:] - bw/2. + 0.5))
    vertices = vertices.at[:,:,1].set(
        spacing * (jnp.arange(bh)[:,None] - bh/2. + 0.5))
    vertices = vertices.at[1:h+1,1:w+1,2].set(height_map)
    
    normals = jnp.zeros((bh, bw, 3), dtype=jnp.float32)
    # va (2, 0, dx)
    lo_x_i = jnp.arange(0,w)
    lo_x = height_map[:,lo_x_i]
    hi_x_i = jnp.arange(2,bw)
    hi_x = height_map[:,hi_x_i]
    dx = (hi_x - lo_x)/2.
    # vb (0, 2, dy)
    lo_y_i = jnp.arange(0,h)
    lo_y = height_map[lo_y_i]
    hi_y_i = jnp.arange(2,bh)
    hi_y = height_map[hi_y_i]
    dy = (hi_y - lo_y)/2.
    # cross product (ya*zb - za*yb), (za*xb - xa*zb), (xa*yb - ya*xb)
    normals = normals.at[1:h+1,1:w+1,0].set(-dx)
    normals = normals.at[1:h+1,1:w+1,1].set(-dy)
    normals = normals.at[:,:,2].set(1)
    normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals.reshape(-1,3)
    
    # border correction (do this after normal computation)
    vertices = vertices.at[ 0,:, 1].add(0.5 * spacing)
    vertices = vertices.at[-1,:, 1].add(-0.5 * spacing)
    vertices = vertices.at[:, 0, 0].add(0.5 * spacing)
    vertices = vertices.at[:,-1, 0].add(-0.5 * spacing)
    
    vertices = vertices.at[ 0, 1:w+1, 2].set(vertices[ 1, 1:w+1, 2])
    vertices = vertices.at[-1, 1:w+1, 2].set(vertices[-2, 1:w+1, 2])
    vertices = vertices.at[ 1:h+1, 0, 2].set(vertices[ 1:h+1, 1, 2])
    vertices = vertices.at[ 1:h+1,-1, 2].set(vertices[ 1:h+1,-2, 2])
    vertices = vertices.at[[0,0,-1,-1],[0,-1,-1,0], 2].set(
        vertices[[1,1,-2,-2],[1,-2,-2,1], 2])
    
    vertices = vertices.reshape(-1,3)
    
    return vertices, normals

def make_height_map_mesh(height_map, spacing=1):
    h, w = height_map.shape
    bh = h + 2
    bw = w + 2
    
    vertices, normals = make_height_map_vertices_and_normals(
        height_map, spacing=spacing)
    
    # | |   |   |   | | #
    uvs = jnp.zeros((bh, bw, 2), dtype=jnp.float32)
    u2 = jnp.linspace(0,1,num=h*2+1)
    u = jnp.concatenate((u2[None,0], u2[1::2], u2[None,-1]))
    v2 = jnp.linspace(0,1,num=h*2+1)
    v = jnp.concatenate((v2[None,0], v2[1::2], v2[None,-1]))
    uvs = uvs.at[:,:,0].set(u[None, :])
    uvs = uvs.at[:,:,1].set(1-v[:, None])
    uvs = uvs.reshape(-1,2)
    
    faces = jnp.zeros((2, bh-1, bw-1, 3), dtype=jnp.int32)
    faces = faces.at[0,:,:,0].add(jnp.arange(0, bw-1)[None, :])
    faces = faces.at[0,:,:,1].add(jnp.arange(1, bw)[None, :])
    faces = faces.at[0,:,:,2].add(jnp.arange(1, bw)[None, :] + bw)
    
    faces = faces.at[1,:,:,0].add(jnp.arange(0, bw-1)[None, :])
    faces = faces.at[1,:,:,1].add(jnp.arange(1, bw)[None, :] + bw)
    faces = faces.at[1,:,:,2].add(jnp.arange(0, bw-1)[None, :] + bw)
    
    faces = faces.at[:,:,:,:].add(jnp.arange(0, bh-1)[:,None,None] * bw)
    faces = faces.reshape(-1,3)
    
    return vertices, normals, uvs, faces
