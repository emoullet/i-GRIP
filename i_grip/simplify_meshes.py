import os
import trimesh as tm
import pyfqmr
from PIL import Image

from i_grip.config import _TLESS_MESH_PATH, _YCVB_MESH_PATH

_TLESS_TARGET_COUNT = 1000
_YCVB_TARGET_COUNT = 3000

for folder in (_TLESS_MESH_PATH, _YCVB_MESH_PATH):
    if folder == _TLESS_MESH_PATH:
        target_count = _TLESS_TARGET_COUNT
    else:
        target_count = _YCVB_TARGET_COUNT
    if not os.path.exists(folder+'_simplified'):
        os.makedirs(folder+'_simplified')
    for file in os.listdir(folder):
        if file.endswith('.ply'):
            mesh = tm.load(os.path.join(folder, file))
            nb_faces = len(mesh.faces)
            mesh_simplifier = pyfqmr.Simplify()
            mesh_simplifier.setMesh(mesh.vertices,mesh.faces)
            mesh_simplifier.simplify_mesh(target_count = int(nb_faces/20), aggressiveness=7, preserve_border=True, verbose=10)
            v, f, n = mesh_simplifier.getMesh()
            new_mesh = tm.Trimesh(vertices=v, faces=f, face_normals=n)
            new_mesh.export(os.path.join(folder+'_simplified', file))
        if file.endswith('.png'):
            file_path = os.path.join(folder, file)
            new_file_path = os.path.join(folder+'_simplified', file)
            os.system(f'cp {file_path} {new_file_path}')