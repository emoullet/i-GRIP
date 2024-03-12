import numpy as np
import trimesh
import time
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import sys  # maxint
import math
import cv2


def minBoundingRect(hull_points_2d):
    #print "Input convex hull points: "
    #print hull_points_2d

    # Compute edges (x2-x1,y2-y1)
    edges = zeros( (len(hull_points_2d)-1,2) ) # empty 2 column array
    for i in range( len(edges) ):
        edge_x = hull_points_2d[i+1,0] - hull_points_2d[i,0]
        edge_y = hull_points_2d[i+1,1] - hull_points_2d[i,1]
        edges[i] = [edge_x,edge_y]
    #print "Edges: \n", edges

    # Calculate edge angles   atan2(y/x)
    edge_angles = zeros( (len(edges)) ) # empty 1 column array
    for i in range( len(edge_angles) ):
        edge_angles[i] = math.atan2( edges[i,1], edges[i,0] )
    #print "Edge angles: \n", edge_angles

    # Check for angles in 1st quadrant
    for i in range( len(edge_angles) ):
        edge_angles[i] = abs( edge_angles[i] % (math.pi/2) ) # want strictly positive answers
    #print "Edge angles in 1st Quadrant: \n", edge_angles

    # Remove duplicate angles
    edge_angles = unique(edge_angles)
    #print "Unique edge angles: \n", edge_angles

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, 99999999999999999999999, 0, 0, 0, 0, 0, 0) # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    for i in range( len(edge_angles) ):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = array([ [ math.cos(edge_angles[i]), math.cos(edge_angles[i]-(math.pi/2)) ], [ math.cos(edge_angles[i]+(math.pi/2)), math.cos(edge_angles[i]) ] ])
        #print "Rotation matrix for ", edge_angles[i], " is \n", R

        # Apply this rotation to convex hull points
        rot_points = dot(R, transpose(hull_points_2d) ) # 2x2 * 2xn
        #print "Rotated hull points are \n", rot_points

        # Find min/max x,y points
        min_x = nanmin(rot_points[0], axis=0)
        max_x = nanmax(rot_points[0], axis=0)
        min_y = nanmin(rot_points[1], axis=0)
        max_y = nanmax(rot_points[1], axis=0)
        #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width*height
        #print "Potential bounding box ", i, ":  width: ", width, " height: ", height, "  area: ", area 

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )
        # Bypass, return the last found rect
        #min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]   
    R = array([ [ math.cos(angle), math.cos(angle-(math.pi/2)) ], [ math.cos(angle+(math.pi/2)), math.cos(angle) ] ])
    #print "Projection matrix: \n", R

    # Project convex hull points onto rotated frame
    proj_points = dot(R, transpose(hull_points_2d) ) # 2x2 * 2xn
    #print "Project hull points are \n", proj_points

    # min/max x,y points are against baseline
    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]
    #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y

    # Calculate center point and project onto rotated frame
    center_x = (min_x + max_x)/2
    center_y = (min_y + max_y)/2
    center_point = dot( [ center_x, center_y ], R )
    #print "Bounding box center point: \n", center_point

    # Calculate corner points and project onto rotated frame
    corner_points = zeros( (4,2) ) # empty 2 column array
    corner_points[0] = dot( [ max_x, min_y ], R )
    corner_points[1] = dot( [ min_x, min_y ], R )
    corner_points[2] = dot( [ min_x, max_y ], R )
    corner_points[3] = dot( [ max_x, max_y ], R )
    #print "Bounding box corner points: \n", corner_points

    #print "Angle of rotation: ", angle, "rad  ", angle * (180/math.pi), "deg"

    return (angle, min_bbox[1], min_bbox[2], min_bbox[3], center_point, corner_points) # rot_angle, area, width, height, center_point, corner_points


def oriented_bounds_2D(points, qhull_options="QbB"):
    """
    Find an oriented bounding box for an array of 2D points.

    Parameters
    ----------
    points : (n,2) float
      Points in 2D.

    Returns
    ----------
    transform : (3,3) float
      Homogeneous 2D transformation matrix to move the
      input points so that the axis aligned bounding box
      is CENTERED AT THE ORIGIN.
    rectangle : (2,) float
       Size of extents once input points are transformed
       by transform
    """
    # create a convex hull object of our points
    # 'QbB' is a qhull option which has it scale the input to unit
    # box to avoid precision issues with very large/small meshes
    convex = ConvexHull(points, qhull_options=qhull_options)

    # (n,2,3) line segments
    hull_edges = convex.points[convex.simplices]
    # (n,2) points on the convex hull
    hull_points = convex.points[convex.vertices]

    # unit vector direction of the edges of the hull polygon
    # filter out zero- magnitude edges via check_valid
    edge_vectors = hull_edges[:, 1] - hull_edges[:, 0]
    edge_norm = np.sqrt(np.dot(edge_vectors**2, [1, 1]))
    edge_nonzero = edge_norm > 1e-10
    edge_vectors = edge_vectors[edge_nonzero] / edge_norm[edge_nonzero].reshape((-1, 1))

    # create a set of perpendicular vectors
    perp_vectors = np.fliplr(edge_vectors) * [-1.0, 1.0]

    # find the projection of every hull point on every edge vector
    # this does create a potentially gigantic n^2 array in memory,
    # and there is the 'rotating calipers' algorithm which avoids this
    # however, we have reduced n with a convex hull and numpy dot products
    # are extremely fast so in practice this usually ends up being fine
    x = np.dot(edge_vectors, hull_points.T)
    y = np.dot(perp_vectors, hull_points.T)

    # reduce the projections to maximum and minimum per edge vector
    bounds = np.column_stack((x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)))
    
    extents = np.diff(bounds.reshape((-1, 2, 2)), axis=1).reshape((-1, 2))
    area = np.prod(extents, axis=1)
    best_idx = area.argmin()
    
    r = np.arctan2(*edge_vectors[best_idx][::-1])

    # find the bounding points
    min_x = x.min(axis=1)
    max_x = x.max(axis=1)
    min_y = y.min(axis=1)
    max_y = y.max(axis=1)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

mesh = trimesh.load('/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/ycbv/models/obj_000004.ply')

# mesh.vertices -= mesh.center_mass

# MESH TF

# transform method can be passed a (4, 4) matrix and will cleanly apply the transform
print('APPLY_TRANSFORM TO MESH')

tf = trimesh.transformations.random_rotation_matrix()
mesh.apply_transform(tf)    
t = time.time()
sc = trimesh.Scene()
sc.add_geometry(trimesh.creation.axis(axis_length=100))
# sc.add_geometry(mesh.bounding_box_oriented)
sc.add_geometry(mesh)
cam_tf = sc.camera_transform
print(f'camera transform: {cam_tf}')
cam_pos = cam_tf[0:3, 3]
print(f'camera position: {cam_pos}')
point_of_view = cam_pos
print(point_of_view.shape)
#get closest point on the mesh to the point of view
t = time.time()
closest_point = mesh.nearest.on_surface([point_of_view])[0].reshape(-1)
print(f'closest point: {closest_point}')
print(f'point of view: {point_of_view}')
print(f'nearest elapsed time: {(time.time() - t)*1000} ms')
# sc.show()

# get normalised vector from the closest point to the point of view
normal = closest_point - point_of_view
print(normal)
normal = normal / np.linalg.norm(normal)
#reshape the normal to be a (3,) vector
normal = normal.reshape(-1)
print(normal.shape)
# get vertices of the mesh in a sphere around the closest point, radius 50, using numpy
vertices = mesh.vertices
print(vertices.shape)

# get the vertices that are in the sphere
sphere_radius = 50
vertices_sphere = vertices - closest_point
vertices_sphere = np.linalg.norm(vertices_sphere, axis=1)
vertices_sphere = vertices_sphere < sphere_radius
vertices_sphere = vertices_sphere.reshape(-1)
vertices_sphere = mesh.vertices[vertices_sphere]
points_sphere = trimesh.points.PointCloud(vertices_sphere,  colors=[0, 255, 0,100])
sc.add_geometry(points_sphere)
sc.add_geometry(trimesh.points.PointCloud([closest_point], colors=[255, 0, 0,255]))
sc.show()
print(vertices.shape)
print(vertices_sphere.shape)


point_of_view = point_of_view.reshape(-1, 1)
t= time.time()
to_2D = trimesh.geometry.plane_transform(origin=point_of_view.T, normal=normal)
# transform mesh vertices to 2D and clip the zero Z
vertices_sphere_2D = trimesh.transform_points(vertices_sphere, to_2D)[:, :2]
vertices_2D = trimesh.transform_points(vertices, to_2D)[:, :2]
closest_point_2D = trimesh.transform_points([closest_point], to_2D)[:, :2]
print(f'prjoection elapsed time: {(time.time() - t)*1000} ms')
t = time.time()


tcv2= time.time()
vertices_sphere_2D = np.array(vertices_sphere_2D, dtype=np.float32).reshape(-1, 1, 2)
hull = cv2.convexHull(vertices_sphere_2D)
rec = cv2.minAreaRect(vertices_sphere_2D)
box = cv2.boxPoints(rec)
# print(vertices_sphere_2D)
print(box)
# rec = box
print(f'elapsed time cv2: {(time.time() - t)*1000} ms')
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