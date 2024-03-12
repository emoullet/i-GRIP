import numpy as np
import trimesh

# attach to logger so trimesh messages will be printed to console
# trimesh.util.attach_to_log()

# mesh objects can be loaded from a file name or from a buffer
# you can pass any of the kwargs for the `Trimesh` constructor
# to `trimesh.load`, including `process=False` if you would like
# to preserve the original loaded data without merging vertices
# STL files will be a soup of disconnected triangles without
# merging vertices however and will not register as watertight
mesh = trimesh.load('/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/ycbv/models_simplified/obj_000004.ply')

# mesh.vertices -= mesh.center_mass

# MESH TF

tf = trimesh.transformations.random_rotation_matrix()
# transform method can be passed a (4, 4) matrix and will cleanly apply the transform
print('APPLY_TRANSFORM TO MESH')
sc = trimesh.Scene()
sc.add_geometry(trimesh.creation.axis(axis_length=100))
print('before apply_transform')
print(mesh.bounding_box_oriented.primitive.extents)
print(mesh.bounding_box_oriented.primitive.transform)
print('bounds')
x_min, x_max = mesh.bounds
y_min, y_max = mesh.bounds
z_min, z_max = mesh.bounds
print(f'mesh.bounds: {mesh.bounds}')
print(f'x_min: {x_min}, x_max: {x_max}')
print(f'y_min: {y_min}, y_max: {y_max}')
print(f'z_min: {z_min}, z_max: {z_max}')
print('max vertices')
x_min, x_max = max(mesh.vertices[:, 0]), min(mesh.vertices[:, 0])
y_min, y_max = max(mesh.vertices[:, 1]), min(mesh.vertices[:, 1])
z_min, z_max = max(mesh.vertices[:, 2]), min(mesh.vertices[:, 2])
print(f'x_min: {x_min}, x_max: {x_max}')
print(f'y_min: {y_min}, y_max: {y_max}')
print(f'z_min: {z_min}, z_max: {z_max}')
print('bounding_box_oriented bounds')
x_min, x_max = mesh.bounding_box_oriented.bounds
y_min, y_max = mesh.bounding_box_oriented.bounds
z_min, z_max = mesh.bounding_box_oriented.bounds
print(f'bounding_box_oriented.bounds: {mesh.bounding_box_oriented.bounds}')
print(f'x_min: {x_min}, x_max: {x_max}')
print(f'y_min: {y_min}, y_max: {y_max}')
print(f'z_min: {z_min}, z_max: {z_max}')
sc.add_geometry(mesh.bounding_box_oriented)
sc.add_geometry(mesh)
sc.show()
mesh.apply_transform(tf)
# mesh.bounding_box_oriented.apply_transform(tf)
print('after apply_transform')
print(mesh.bounding_box_oriented.primitive.extents)
print(mesh.bounding_box_oriented.primitive.transform)
sc.add_geometry(mesh)
sc.add_geometry(mesh.bounding_box_oriented)
sc.show()

print('APPLY_TRANSFORM TO SCENE')

mesh = trimesh.load('/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/ycbv/models_simplified/obj_000004.ply')
sc = trimesh.Scene()
sc.add_geometry(trimesh.creation.axis(axis_length=100)) 
print(mesh.bounding_box_oriented.primitive.extents)
print(mesh.bounding_box_oriented.primitive.transform)
# mesh.apply_transform(tf)
# mesh.bounding_box_oriented.apply_transform(tf)
sc.add_geometry(mesh, geom_name='mesh')
sc.graph.update('mesh', matrix=tf, geometry='mesh')
print(mesh.bounding_box_oriented.primitive.extents)
print(mesh.bounding_box_oriented.primitive.transform)
sc.show()

print('APPLY_TRANSFORM TO MESH AFTER SCENE')

sc = trimesh.Scene()
sc.add_geometry(trimesh.creation.axis(axis_length=100))
print('before apply_transform')
print(mesh.bounding_box_oriented.primitive.extents)
print(mesh.bounding_box_oriented.primitive.transform)
# mesh.bounding_box_oriented.apply_transform(tf)
sc.add_geometry(mesh)
sc.show()
mesh.apply_transform(tf)
print('after apply_transform')
print(mesh.bounding_box_oriented.primitive.extents)
print(mesh.bounding_box_oriented.primitive.transform)
sc.show()
# (mesh + mesh.bounding_sphere).show()

# bounding spheres and bounding cylinders of meshes are also
# available, and will be the minimum volume version of each
# except in certain degenerate cases, where they will be no worse
# than a least squares fit version of the primitive.
print(mesh.bounding_box_oriented.volume,
      mesh.bounding_cylinder.volume,
      mesh.bounding_sphere.volume)