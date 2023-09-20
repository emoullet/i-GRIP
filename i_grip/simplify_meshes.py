import os
import trimesh as tm
import pyfqmr
from PIL import Image


_TLESS_MESH_PATH = '/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/tless/models_cad'
_YCVB_MESH_PATH = '/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/ycbv/models'

for folder in (_TLESS_MESH_PATH, _YCVB_MESH_PATH):
    if folder == _TLESS_MESH_PATH:
        target_count = 1000
    else:
        target_count = 3000
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
            # uv = mesh.visual.uv
            # name = file.split('.')[0]
            # im = Image.open(os.path.join(folder,file+".png"))
            # material = tm.visual.texture.SimpleMaterial(image=im)
            # color_visuals = tm.visual.TextureVisuals(uv=uv, image=im, material=material)
            new_mesh = tm.Trimesh(vertices=v, faces=f, face_normals=n)
            # new_mesh.visual = color_visuals
            new_mesh.export(os.path.join(folder+'_simplified', file))
        if file.endswith('.png'):
            #copy file
            file_path = os.path.join(folder, file)
            new_file_path = os.path.join(folder+'_simplified', file)
            os.system(f'cp {file_path} {new_file_path}')