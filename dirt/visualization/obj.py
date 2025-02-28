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
