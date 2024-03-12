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

mesh = trimesh.load('/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/ycbv/models_simplified/obj_000004.ply')

# mesh.vertices -= mesh.center_mass

# MESH TF

# transform method can be passed a (4, 4) matrix and will cleanly apply the transform
print('APPLY_TRANSFORM TO MESH')

tf = trimesh.transformations.random_rotation_matrix()
mesh.apply_transform(tf)    
t = time.time()
sc = trimesh.Scene()
sc.add_geometry(trimesh.creation.axis(axis_length=100))
sc.add_geometry(mesh.bounding_box)
sc.add_geometry(mesh)
point_of_view = [0, 0, -100]
# sc.show()
rpad = 1
normal = [0, 1, 0]
origin = [0, 100, 0]
t = time.time()
outline = mesh.projected(normal=normal, origin=origin, rpad=rpad, max_regions = 100)
print(f'elapsed time: {(time.time() - t)*1000} ms')
# outline.show()

t= time.time()
to_2D = trimesh.geometry.plane_transform(origin=origin, normal=normal)
# transform mesh vertices to 2D and clip the zero Z
vertices_2D = trimesh.transform_points(mesh.vertices, to_2D)[:, :2]
print(f'prjoection elapsed time: {(time.time() - t)*1000} ms')
t = time.time()

# tf, extents, rec = trimesh.bounds.oriented_bounds_2D(vertices_2D)
# print(tf)
# print(rec)
# print(f'elapsed time trimesh oriented: {(time.time() - t)*1000} ms')
# fig, ax = plt.subplots()
# ax.plot(vertices_2D[:, 0], vertices_2D[:, 1], 'o')
# # rec = [r/2 for r in rec]
# # apply the transform to the rectangle
# p1 = [rec[0], rec[1]]
# p2 = [rec[0], rec[3]]
# p3 = [rec[1], rec[1]]
# p4 = [rec[1], rec[3]]
# # rec = trimesh.transform_points([p1, p2, p3, p4], tf)
# print(rec)
# ax.plot([p1[0], p2[0], p3[0], p4[0], p1[0]], [p1[1], p2[1], p3[1], p4[1], p1[1]])
# # ax.plot(rec[:, 0], rec[:, 1])
# plt.show()

# #find the barycenter of the 2D vertices
# barycenter = np.mean(vertices_2D, axis=0)
# area = 0
# best_angle = 0
# # rotate all vertices incrementally to find the best orientation which minimize the maxium x and y values across all vertices
# for angle in range(0, 360, 1):
#     angle = np.deg2rad(angle)
#     R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
#     vertices_2D_rotated = np.dot(vertices_2D - barycenter, R) + barycenter
#     max_x_rotated = np.max(vertices_2D_rotated[:, 0])
#     max_y_rotated = np.max(vertices_2D_rotated[:, 1])
#     area_rotated = max_x_rotated * max_y_rotated
#     if area_rotated < area:
#         best_angle = angle
# print(f'best angle: {np.rad2deg(best_angle)}')
# print(f'elapsed time: {(time.time() - t)*1000} ms')

# t = time.time()


# # Calcul du centre de gravité
# barycenter = np.mean(vertices_2D, axis=0)

# # Centrer les sommets par rapport au centre de gravité
# vertices_centered = vertices_2D - barycenter

# # Création de la grille d'angles en radians
# angles = np.deg2rad(np.arange(0, 360, 1))

# # Création de la matrice de rotation pour tous les angles
# cos_theta = np.cos(angles)
# sin_theta = np.sin(angles)
# R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

# # Rotation de tous les sommets pour tous les angles
# vertices_rotated = np.dot(vertices_centered, R) 

# # Calcul des dimensions maximales pour chaque angle
# max_x_rotated = np.max(vertices_rotated[:, :, 0], axis=0)
# max_y_rotated = np.max(vertices_rotated[:, :, 1], axis=0)

# # Calcul des aires pour chaque angle
# areas_rotated = max_x_rotated * max_y_rotated

# # Trouver l'angle avec la plus petite aire
# best_angle_index = np.argmin(areas_rotated)
# best_angle = angles[best_angle_index]
# area = areas_rotated[best_angle_index]
# print(f'best angle: {np.rad2deg(best_angle)}')
# print(f'best area: {area}')
# print(f'elapsed time: {(time.time() - t)*1000} ms')

# # plot the vertices and the rectangle
# fig, ax = plt.subplots()
# ax.plot(vertices_2D[:, 0], vertices_2D[:, 1], 'o')
# ax.plot(vertices_rotated[:, best_angle_index, 0], vertices_rotated[:, best_angle_index, 1], 'o')
# ax.set_aspect('equal')
# plt.show()

# t= time.time()
# rec = minimum_bounding_rectangle(vertices_2D)
# print(rec)
# print(f'elapsed time other: {(time.time() - t)*1000} ms')
# fig, ax = plt.subplots()
# ax.plot(vertices_2D[:, 0], vertices_2D[:, 1], 'o')
# ax.plot([rec[0, 0], rec[1, 0], rec[2, 0], rec[3, 0], rec[0, 0]], [rec[0, 1], rec[1, 1], rec[2, 1], rec[3, 1], rec[0, 1]])
# plt.show()


# t= time.time()
# rec = oriented_bounds_2D(vertices_2D)
# print(rec)
# print(f'elapsed time other: {(time.time() - t)*1000} ms')
# fig, ax = plt.subplots()
# ax.plot(vertices_2D[:, 0], vertices_2D[:, 1], 'o')
# ax.plot([rec[0, 0], rec[1, 0], rec[2, 0], rec[3, 0], rec[0, 0]], [rec[0, 1], rec[1, 1], rec[2, 1], rec[3, 1], rec[0, 1]])
# plt.show()



t= time.time()
vertices_2D = np.array(vertices_2D, dtype=np.float32)
vertices_2D = vertices_2D.reshape(-1, 1, 2)
hull = cv2.convexHull(vertices_2D)
rec = cv2.minAreaRect(vertices_2D)
box = cv2.boxPoints(rec)
print(vertices_2D)
print(box)
# rec = box
print(f'elapsed time cv2: {(time.time() - t)*1000} ms')
fig, ax = plt.subplots()
ax.plot(vertices_2D[:,:, 0], vertices_2D[:,:, 1], 'o')
 # add box first point to close the rectangle
box = np.vstack([box, box[0]])
ax.plot(box[:, 0], box[:, 1])
plt.show()