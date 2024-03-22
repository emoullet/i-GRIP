import os
import trimesh as tm
import pyfqmr
from PIL import Image
import argparse
from i_grip.config import _TLESS_MESH_PATH, _YCVB_MESH_PATH

DEFAULT_FACES_FACTOR = 30
# DEFAULT_FACES_FACTOR = 20 softer mesh simplification

def simplify_meshes(faces_factor=20, suffix=''):
    for folder in (_TLESS_MESH_PATH, _YCVB_MESH_PATH):
        folder = str(folder)
        if not os.path.exists(folder+suffix):
            os.makedirs(folder+suffix)
        for file in os.listdir(folder):
            if file.endswith('.ply'):
                mesh = tm.load(os.path.join(folder, file))
                nb_faces = len(mesh.faces)
                mesh_simplifier = pyfqmr.Simplify()
                mesh_simplifier.setMesh(mesh.vertices,mesh.faces)
                mesh_simplifier.simplify_mesh(target_count = int(nb_faces/faces_factor), aggressiveness=5, preserve_border=True, verbose=10)
                v, f, n = mesh_simplifier.getMesh()
                new_mesh = tm.Trimesh(vertices=v, faces=f, face_normals=n)
                new_mesh.export(os.path.join(folder+suffix, file))
            if file.endswith('.png'):
                file_path = os.path.join(folder, file)
                new_file_path = os.path.join(folder+suffix, file)
                os.system(f'cp {file_path} {new_file_path}')
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces_factor', type=int, default=DEFAULT_FACES_FACTOR)
    parser.add_argument('--suffix', type=str, default='_simplified')
    args = parser.parse_args()
    simplify_meshes(args.faces_factor, args.suffix)