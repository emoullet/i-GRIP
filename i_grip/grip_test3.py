import numpy as np
import trimesh
import time
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import cv2


from pysdf import SDF


def scene_callback(sc):
    cam_tf = sc.camera_transform
    
    vdir = np.array([0, 0, 1,1])    
    p2 = np.array([0, 0, -100,1])
    
    point_of_view = cam_tf @ p2
    point_of_view = point_of_view[:3]
    ray_direction = (cam_tf @ vdir)[:3]
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    ray_viz = trimesh.load_path(np.array([point_of_view, point_of_view - ray_direction * 1000]))
    sc.add_geometry(ray_viz, geom_name='ray_viz')
    # sc.delete_geometry('ray_viz')
    # print(point_of_view.shape)
    #create 8 points forming a cube around the point of view
    delta = 10
    cube_points = [[point_of_view[0] + delta, point_of_view[1] + delta, point_of_view[2] + delta],
                    [point_of_view[0] + delta, point_of_view[1] + delta, point_of_view[2] - delta],
                    [point_of_view[0] + delta, point_of_view[1] - delta, point_of_view[2] + delta],
                    [point_of_view[0] + delta, point_of_view[1] - delta, point_of_view[2] - delta],
                    [point_of_view[0] - delta, point_of_view[1] + delta, point_of_view[2] + delta],
                    [point_of_view[0] - delta, point_of_view[1] + delta, point_of_view[2] - delta],
                    [point_of_view[0] - delta, point_of_view[1] - delta, point_of_view[2] + delta],
                    [point_of_view[0] - delta, point_of_view[1] - delta, point_of_view[2] - delta]]
    # make list of 8 np.array
    cube_points = [np.array(p) for p in cube_points]
    
    #get closest point on the mesh to the point of view !
    t = time.time()
    closest_point = mesh.nearest.on_surface([point_of_view])[0].reshape(-1)
    closest_points = mesh.nearest.on_surface([point_of_view]+cube_points)
    # print(f'closest point: {closest_point}')
    # print(f'point of view: {point_of_view}')
    # print(f'nearest elapsed time: {(time.time() - t)*1000} ms')
    
    # perform a ray trace in the direction of the camera
    ray_origin = point_of_view
    p2 = ray_origin 
    # p2sphere= trimesh.primitives.Sphere(radius=10, center=p2)
    # p2sphere.visual.face_colors = [0, 0, 255, 255]
    # sc.delete_geometry('p2') 
    # sc.add_geometry(p2sphere, geom_name='p2')
    locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=[ray_origin], ray_directions = [-ray_direction*1000], multiple_hits=False)
    print(locations)
    if len(locations) > 0:
        impact_point = locations[0]    
        # choose point
        chosen_point = impact_point
        # print(f'impact point: {impact_point}')
    # else:
    chosen_point = impact_point

    # get normalised vector from the closest point to the point of view
    normal = chosen_point - point_of_view
    # print(normal)
    normal = normal / np.linalg.norm(normal)
    #reshape the normal to be a (3,) vector
    normal = normal.reshape(-1)
    # print(normal.shape)
    # get vertices of the mesh in a sphere around the closest point, radius 50, using numpy
    vertices = mesh.vertices
    # print(vertices.shape)

    # get the vertices that are in the sphere
    sphere_radius = 50
    vertices_sphere = vertices - chosen_point
    vertices_sphere = np.linalg.norm(vertices_sphere, axis=1)
    vertices_sphere = vertices_sphere < sphere_radius
    vertices_sphere = vertices_sphere.reshape(-1)
    vertices_sphere = mesh.vertices[vertices_sphere]
    points_sphere = trimesh.points.PointCloud(vertices_sphere,  colors=[0, 255, 0,100])
    sc.delete_geometry('sphere')
    sc.delete_geometry('chosen_point')
    sc.add_geometry(points_sphere, geom_name='sphere')
    sc.add_geometry(trimesh.points.PointCloud([chosen_point], colors=[255, 0, 0,255]), geom_name='chosen_point')
    sc.delete_geometry('ray')
    sc.add_geometry(ray_viz, geom_name='ray')
    # print(vertices.shape)
    # print(vertices_sphere.shape)
    
    
    point_of_view = point_of_view.reshape(-1, 1)
    t= time.time()
    to_2D = trimesh.geometry.plane_transform(origin=point_of_view.T, normal=normal)
    # transform mesh vertices to 2D and clip the zero Z
    vertices_sphere_2D = trimesh.transform_points(vertices_sphere, to_2D)[:, :2]
    vertices_2D = trimesh.transform_points(vertices, to_2D)[:, :2]
    chosen_point_2D = trimesh.transform_points([chosen_point], to_2D)[:, :2]
    # print(f'prjoection elapsed time: {(time.time() - t)*1000} ms')
    t = time.time()


    tcv2= time.time()
    vertices_sphere_2D = np.array(vertices_sphere_2D, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(vertices_sphere_2D)
    rec = cv2.minAreaRect(vertices_sphere_2D)
    width = rec[1][0]
    height = rec[1][1]
    small_side = min(width, height)
    big_side = max(width, height)
    ratio = small_side / big_side
    small_threshold = 0.7*2
    ratio_threshold = 0.8
    if small_side > small_threshold*sphere_radius:
        res = 'too big'
    elif ratio < ratio_threshold:
        res = 'palmar'
    else:
        res = 'pinch'
    print(f'small side: {small_side}, big side: {big_side}, ratio: {ratio}, res: {res}')
    
    box = cv2.boxPoints(rec)
    # print(f'elapsed time cv2: {(time.time() - t)*1000} ms')
    # print(vertices_sphere_2D)
    # print(f'rec : {rec}')
    # print(f'box : {box}\n')

mesh = trimesh.load('/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/ycbv/models/obj_000005.ply')

# mesh.vertices -= mesh.center_mass

# MESH TF

# transform method can be passed a (4, 4) matrix and will cleanly apply the transform
print('APPLY_TRANSFORM TO MESH')

# tf = trimesh.transformations.random_rotation_matrix()
# mesh.apply_transform(tf)    

signed_distance_finder = SDF(mesh.vertices, mesh.faces)
t = time.time()
sc = trimesh.Scene()
sc.add_geometry(trimesh.creation.axis(axis_length=100))

# sc.camera_transform = trimesh.transformations.rotation_matrix(np.pi, np.array([0,1,0]), np.array([0,0,0]))
# cam = trimesh.creation.camera_marker(sc.camera, marker_height = 30)
# sc.add_geometry(mesh.bounding_box_oriented)
sc.add_geometry(mesh)
# sc.add_geometry(cam)
sc.show(callback=scene_callback,callback_period=1./30, line_settings={'point_size':20}, start_loop=True, visible=    True, viewer='gl')



# rec = box
fig, ax = plt.subplots()
ax.plot(vertices_2D[:, 0], vertices_2D[:,1], 'o')
ax.plot(vertices_sphere_2D[:,:, 0], vertices_sphere_2D[:,:, 1], 'o')
ax.plot(closest_point_2D[:, 0], closest_point_2D[:, 1], 'o')
 # add box first point to close the rectangle
box = np.vstack([box, box[0]])
ax.plot(box[:, 0], box[:, 1])
plt.show()
#create a 3D convex mesh from sphere vertices
mesh_sphere = trimesh.PointCloud(vertices_sphere)
#compute the convex hull of the sphere vertices
hull = mesh_sphere.convex_hull
hull.show()
mesh_sphere.show()