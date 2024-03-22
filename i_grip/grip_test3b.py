import numpy as np
import trimesh
import time
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import cv2
from i_grip.clean_scene import CleanScene


from pysdf import SDF

class mySDF(SDF):
    def __init__(self, vertices, faces):
        super().__init__(vertices, faces)
        self.vertices = vertices
        self.faces = faces
    def __call__(self, points):
        return super().__call__(points)
    def __getitem__(self, key):
        return super().__getitem__(key)
    def __len__(self):
        return super().__len__()
    

def in_ellipsoid(points, trans_mat, l1, l2, l3):
    # Ajouter une colonne de 1 à la matrice de points pour étendre en notation affine
    P = np.hstack((points, np.ones((points.shape[0], 1))))
    # Transformer les points en coordonnées locales de l'ellipsoïde
    points_local = (trans_mat @ P.T).T
    # Vérifier si les points sont dans l'ellipsoïde
    return (points_local[:, 0] / l1) ** 2 + (points_local[:, 1] / l2) ** 2 + (points_local[:, 2] / l3) ** 2 <= 1

# de
def draw_ellipsoid(l1, l2, l3, R, chosen_point, n_points=1000):
    # Générer des angles uniformément distribués sur la sphère unité
    theta = np.random.uniform(0, np.pi, n_points)
    phi = np.random.uniform(0, 2*np.pi, n_points)

    # Calculer les coordonnées sphériques des points
    r = np.random.uniform(0, 1, n_points)**(1/3)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Transformer les points en coordonnées locales de l'ellipsoïde
    points_local = np.stack((x, y, z), axis=1)
    points_local[:, 0] *= l1
    points_local[:, 1] *= l2
    points_local[:, 2] *= l3
    points_local = R @ points_local.T
    points_local = points_local.T + chosen_point

    return points_local

def scene_callback(sc):
    cam_tf = sc.camera_transform
    
    vdir = np.array([0, 0, 1,1])    
    p2= np.array([0, 0, -50,1])
    pc= np.array([0, 0, -50,1])
    range_z = (-75, 25)
    deltas = (-5,5)
    nb_p2 = 2
    if nb_p2 == 1:
        p2s = [p2]
    else:
        p2s = [np.array([0, 0, z, 1]) for z in np.linspace(range_z[0], range_z[1], nb_p2)]
        p2s = [pc + np.array([dx, dy, dz, 0]) for dx in np.linspace(deltas[0], deltas[1], nb_p2) for dy in np.linspace(deltas[0], deltas[1], nb_p2) for dz in np.linspace(range_z[0], range_z[1], nb_p2)]
    
    # print(f'p2s: {p2s}')
    
    point_of_view = cam_tf @ p2
    point_of_view = point_of_view[:3]
    point_of_views = [(cam_tf @ p2)[:3] for p2 in p2s]
    
    # point_of_view = cam_tf[:3,3]
    # point_of_views = [point_of_view]
    
    ray_direction = (cam_tf @ vdir)[:3]
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    ray_viz = trimesh.load_path(np.array([point_of_view, point_of_view - ray_direction * 1000]))
    sc.delete_geometry('ray_viz')
    sc.add_geometry(ray_viz, geom_name='ray_viz')
    
    multi_point=True
    cube_points = []
    circle_points = []
    sphere_points = []
    for i, point_of_view in enumerate(point_of_views):
        if multi_point:
            # CUBE
            delta = 10
            cube_points_l = [[point_of_view[0] + delta, point_of_view[1] + delta, point_of_view[2] + delta],
                            [point_of_view[0] + delta, point_of_view[1] + delta, point_of_view[2] - delta],
                            [point_of_view[0] + delta, point_of_view[1] - delta, point_of_view[2] + delta],
                            [point_of_view[0] + delta, point_of_view[1] - delta, point_of_view[2] - delta],
                            [point_of_view[0] - delta, point_of_view[1] + delta, point_of_view[2] + delta],
                            [point_of_view[0] - delta, point_of_view[1] + delta, point_of_view[2] - delta],
                            [point_of_view[0] - delta, point_of_view[1] - delta, point_of_view[2] + delta],
                            [point_of_view[0] - delta, point_of_view[1] - delta, point_of_view[2] - delta]]
            cube_points += [np.array(p) for p in cube_points_l]
            
            # CIRCLE
            nb_points = 10
            disk_radiuses = [(i)*3 for i in range(1, 5)]
            circle_points_l = [point_of_view]
            for disk_radius in disk_radiuses:
                circle_points_l += [[point_of_view[0] + disk_radius*np.cos(2*np.pi/nb_points*i), point_of_view[1] + disk_radius*np.sin(2*np.pi/nb_points*i), point_of_view[2]] for i in range(nb_points)]
            circle_points += [np.array(p) for p in circle_points_l]
            
            # SPHERE
            sphere_radiuses = [5]
            nb_angles = 10
            sphere_points_l = [point_of_view]
            for sphere_radius in sphere_radiuses:
                sphere_points_l += [[point_of_view[0] + sphere_radius*np.sin(theta)*np.cos(phi), point_of_view[1] + sphere_radius*np.sin(theta)*np.sin(phi), point_of_view[2] + sphere_radius*np.cos(theta)] for theta in np.linspace(0, np.pi, int(nb_angles/2)) for phi in np.linspace(0, 2*np.pi, nb_angles)]
            sphere_points += [np.array(p) for p in sphere_points_l]
    point_of_views_geom = [trimesh.primitives.Sphere(radius=1, center=point_of_view) for point_of_view in point_of_views]
    for i, point_of_viewg in enumerate(point_of_views_geom):
        point_of_viewg.visual.face_colors = [0, 0, 200, 100]
        sc.delete_geometry(f'point_of_view_{i}')
        sc.add_geometry(point_of_viewg, geom_name=f'point_of_view_{i}')
    sphere_points_geom = trimesh.points.PointCloud(sphere_points, colors=[22, 100, 100,255])
    sc.delete_geometry('sphere_points')
    sc.add_geometry(sphere_points_geom, geom_name='sphere_points')
    # print(f'len circle_points: {len(circle_points)}')
    # print(f'len sphere_points: {len(sphere_points)}')
    points_mode = 'sphere'
    if points_mode == 'circle':
        origins = circle_points
        ray_directions = [-ray_direction*1000 for i in range(len(circle_points))]
    elif points_mode == 'sphere':
        origins = sphere_points
        ray_directions = [-ray_direction*1000 for i in range(len(sphere_points))]
    else:
        origins = [point_of_view]
        ray_directions = [-ray_direction*1000]

    locations=[]
    if False:
        t = time.time()
        locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=origins, ray_directions = ray_directions, multiple_hits=False)
        print(f'ray trace elapsed time: {(time.time() - t)*1000} ms')
        sc.delete_geometry('impact_viz')
        sc.add_geometry(trimesh.points.PointCloud(locations, colors=[0, 0, 255,255]), geom_name='impact_viz')
        
        print(type(locations))
    
    # t = time.time()
    # closest_points, _, _ = mesh.nearest.on_surface(origins)
    # print(f'nearest elapsed time: {(time.time() - t)*1000} ms')
    t = time.time()
    closest_points_indexes = signed_distance_finder.nn(origins)
    closest_points = signed_distance_finder.vertices[closest_points_indexes]
    print(f'signed distance elapsed time: {(time.time() - t)*1000} ms')
    # print(f'closest_points: {closest_points}')
    # closest_points = np.array(closest_points).reshape(-1, 3)
    # print(f'type closest_points: {type(closest_points)}')
    sc.delete_geometry('closest_points')
    sc.add_geometry(trimesh.points.PointCloud(closest_points, colors=[0, 255, 255,255]), geom_name='closest_points')
    if len(locations) > 0:
        impact_point = np.mean(locations, axis=0)  
        print(f'impact point: {impact_point}')
    closest_point = np.mean(closest_points, axis=0)
    # choose point
    chosen_point = closest_point
    # else:
    chosen_point = closest_point
    
    
    chosen_point_geom = trimesh.primitives.Sphere(radius=10, center=chosen_point)
    chosen_point_geom.visual.face_colors = [255, 0, 0, 255]
    sc.delete_geometry('chosen_point')
    sc.add_geometry(chosen_point_geom, geom_name='chosen_point')
    
    # get normalised vector from the closest point to the point of view
    vdirdir = chosen_point - point_of_view
    vdirdir = vdirdir / np.linalg.norm(vdirdir)
    
    # sc.delete_geometry('ray_viz')
    
    
    # get vertices of the mesh in a sphere around the closest point, radius 50, using numpy
    vertices = mesh.vertices
    # print(f'nb vertices: {len(vertices)}')
    # get the vertices that are in the sphere
    sphere_radius = 50
    vertices_sphere = vertices - chosen_point
    vertices_sphere = np.linalg.norm(vertices_sphere, axis=1)
    vertices_sphere = vertices_sphere < sphere_radius
    vertices_sphere = vertices_sphere.reshape(-1)
    vertices_sphere = mesh.vertices[vertices_sphere]
    points_sphere = trimesh.points.PointCloud(vertices_sphere,  colors=[0, 255, 0,100])
    sc.delete_geometry('sphere')
    sc.add_geometry(points_sphere, geom_name='sphere')
    

    
    
    # Définir les paramètres de l'ellipsoïde
    l1, l2, l3 = 5, 50, 50 # longueurs des axes

    # Définir la matrice de rotation pour aligner l'axe principal avec l'axe des x
    
    
    ellipsoid = False
    if ellipsoid:
        
        a = np.array([0,0,1])
        v = vdirdir/np.linalg.norm(vdirdir)
        # v = a/np.linalg.norm(a)
        v_geom = trimesh.load_path(np.array([chosen_point, chosen_point - v*1000]))
        sc.delete_geometry('v_geom')
        sc.add_geometry(v_geom, geom_name='v_geom')
        # v = a/np.linalg.norm(a)
        u = np.array([0, 0, 1])
        if np.abs(np.dot(v, u)) > 1e-5:
            u = np.array([0, 1, 0])
        w = np.cross(v, u)
        w= w/np.linalg.norm(w)
        u = np.cross(w, v)
        u = u/np.linalg.norm(u)
        R = np.stack((v, u, w), axis=0)
        ellipsoid_points = draw_ellipsoid(l1, l2, l3, R, chosen_point, n_points=10000)
        ellipsoid_points_geom = trimesh.points.PointCloud(ellipsoid_points,  colors=[255, 0, 0,100])
        sc.delete_geometry('ellipsoid_points')
        sc.add_geometry(ellipsoid_points_geom, geom_name='ellipsoid_points')

        # Définir la matrice de transformation pour transformer les points en coordonnées locales de l'ellipsoïde
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ chosen_point
        vertices_in_ellipsoid = mesh.vertices[in_ellipsoid(vertices, T, l1, l2, l3)]
        vertices_in_ellipsoid_geom = trimesh.points.PointCloud(vertices_in_ellipsoid,  colors=[255, 0, 0,100])
        sc.delete_geometry('vertices_in_ellipsoid')
        sc.add_geometry(vertices_in_ellipsoid_geom, geom_name='vertices_in_ellipsoid')
        
    point_of_view = point_of_view.reshape(-1, 1)
    t= time.time()
    to_2D = trimesh.geometry.plane_transform(origin=point_of_view.T, normal=vdirdir)
    # transform mesh vertices to 2D and clip the zero Z
    vertices_sphere_2D = trimesh.transform_points(vertices_sphere, to_2D)[:, :2]
    # vertices_sphere_2D = trimesh.transform_points(vertices_in_ellipsoid, to_2D)[:, :2]
    vertices_2D = trimesh.transform_points(vertices, to_2D)[:, :2]
    chosen_point_2D = trimesh.transform_points([chosen_point], to_2D)[:, :2]
    # print(f'prjoection elapsed time: {(time.time() - t)*1000} ms')
    t = time.time()


    tcv2= time.time()
    vertices_sphere_2D = np.array(vertices_sphere_2D, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(vertices_sphere_2D)
    rec = cv2.minAreaRect(vertices_sphere_2D)
    rec = cv2.fitEllipse(vertices_sphere_2D)
    # rec = cv2.fitEllipseAMS(vertices_sphere_2D)
    width = rec[1][0]
    height = rec[1][1]
    small_side = min(width, height)
    big_side = max(width, height)
    ratio = small_side / big_side
    small_threshold = 0.7*2*sphere_radius
    big_threshold = 0.4*2*sphere_radius
    ratio_threshold = 0.7
    if small_side > small_threshold:
        res = 'too big'
        cause = 'small threshold'
    elif big_side < big_threshold:
        res = 'pinch'
        cause = 'big threshold'
    elif ratio < ratio_threshold:
        res = 'palmar'
        cause = 'ratio'
    else:
        res = 'pinch'
        cause = 'else'
    print(f'small side: {small_side}, big side: {big_side}, ratio: {ratio}, small_threshold: {small_threshold}, big_threshold: {big_threshold}, res: {res}, cause: {cause}')

mesh = trimesh.load('/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/ycbv/models_simplified_soft/obj_000005.ply')
mesh = trimesh.load('/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/ycbv/models/obj_000004.ply')

# mesh.vertices -= mesh.center_mass

# MESH TF

# transform method can be passed a (4, 4) matrix and will cleanly apply the transform
print('APPLY_TRANSFORM TO MESH')

# tf = trimesh.transformations.random_rotation_matrix()
# mesh.apply_transform(tf)    

signed_distance_finder = SDF(mesh.vertices, mesh.faces)
t = time.time()
sc = CleanScene()
# sc.add_geometry(trimesh.creation.axis(axis_length=100))

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