import jax.numpy as jnp

from dirt.gridworld2d.observations import local_view

def audio(
    speaker_x,
    speaker_data,
    listener_x,
    #listener_r,
    world_size,
    cell_size,
    noise,
):
    # get shapes and dtypes
    ns, c = speaker_data.shape
    nl, d = listener_x.shape
    assert d == 2
    float_dtype = speaker_data.dtype
    grid_size = world_size // cell_size + 1
    
    # build the sound map
    sound_map = jnp.zeros((*grid_size, c), dtype=float_dtype)
    speaker_cells = speaker_x // cell_size
    speaker_uv = (speaker_x % cell_size).astype(float_dtype) / cell_size
    speaker_uv_inv = 1. - speaker_uv
    sound_map = sound_map.at[speaker_cells[...,0],speaker_cells[...,1]].add(
        speaker_data * speaker_uv_inv[...,0,None] * speaker_uv_inv[...,1,None])
    sound_map = sound_map.at[speaker_cells[...,0]+1,speaker_cells[...,1]].add(
        speaker_data * speaker_uv[...,0,None] * speaker_uv_inv[...,1,None])
    sound_map = sound_map.at[speaker_cells[...,0],speaker_cells[...,1]+1].add(
        speaker_data * speaker_uv_inv[...,0,None] * speaker_uv[...,1,None])
    sound_map = sound_map.at[speaker_cells[...,0]+1,speaker_cells[...,1]+1].add(
        speaker_data * speaker_uv[...,0,None] * speaker_uv[...,1,None])
    
    print(sound_map[:,:,0])
    
    # query the sound map
    listener_cells = listener_x // cell_size
    listener_samples = local_view(
        listener_cells,
        jnp.zeros((nl,), dtype=jnp.int32),
        sound_map,
        ((0,0),(2,2)),
    )
    
    # interpolate the values
    listener_uv = (listener_x % cell_size).astype(float_dtype) / cell_size
    listener_uv_inv = 1. - listener_uv
    listener_00 = listener_uv_inv[...,0,None] * listener_uv_inv[...,1,None]
    listener_01 = listener_uv_inv[...,0,None] * listener_uv[...,1,None]
    listener_10 = listener_uv[...,0,None] * listener_uv_inv[...,1,None]
    listener_11 = listener_uv[...,0,None] * listener_uv[...,1,None]
    sampled_audio = (
        listener_samples[:,0,0] * listener_00 +
        listener_samples[:,0,1] * listener_01 +
        listener_samples[:,1,0] * listener_10 +
        listener_samples[:,1,1] * listener_11
    )
    
    return sampled_audio

if __name__ == '__main__':
    speaker_x = jnp.array([
        [2,2],
        [4,4],
        [6,6],
    ])
    speaker_data = jnp.array([
        [1],
        [10],
        [100],
    ]).astype(jnp.bfloat16)
    listener_x = jnp.array([
        [0,0],
        [4,4],
        [6,6],
    ])
    
    a = audio(
        speaker_x,
        speaker_data,
        listener_x,
        jnp.array([8,8], dtype=jnp.int32),
        4,
        0.,
    )
    
    print(a)
