import trimesh as tm
import numpy as np

def scene_callback(sc):
    p2 = np.array([0, 0, -100,1])
    vdir = np.array([0, 0, 1,1])
    cam_tf = sc.camera_transform
    cam_res = sc.camera.resolution
    cam_rot = cam_tf[:3,:3]
    ray_dir = (cam_tf @ vdir)[:3]
    # ray_dir = cam_rot @ ray_dir[:3]
    print(f'norm : {np.linalg.norm(ray_dir)}')
    print(f'ray dir : {ray_dir}')
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    p2 = cam_tf @ p2
    p2 = p2[:3]
    ray_viz = tm.load_path(np.array([p2, p2 - ray_dir * 1000]))
    # sc.delete_geometry('ray_viz')
    sc.add_geometry(ray_viz, geom_name='ray_viz')
    cam_pos = cam_tf[:3,3]
    print(f'pos tf : {cam_pos}')
    locations, index_ray, index_tri = sphere1.ray.intersects_location(ray_origins=[p2], ray_directions = [-ray_dir], multiple_hits=False)
    if len(locations) > 0:
        impact_point = locations[0]
        impact_viz = tm.primitives.Sphere(radius=10, center=impact_point)
        impact_viz.visual.face_colors = [255, 0, 0, 255]
        sc.delete_geometry('impact_viz')    
        sc.add_geometry(impact_viz, geom_name='impact_viz')
    # cam_pos = sc.camera.coordinates
    # print(f'pos : {cam_pos}')
    # p2 = cam_pos + (cam_tf @ np.array([cam_res[0], -cam_res[1], 0,1]))[:3]
    
    sphere2 = tm.primitives.Sphere(radius=1, center=p2)
    sphere2.visual.face_colors = [0, 0, 255, 255]
    # sc.delete_geometry('sphere2')
    # sc.add_geometry(sphere2, geom_name='sphere2')
    
sc = tm.Scene()
sphere1 = tm.load('/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/ycbv/models/obj_000005.ply')
# sphere1 = tm.primitives.Sphere(radius=50)
sphere1.visual.face_colors = [0, 255, 0, 255]
sc.add_geometry(sphere1, geom_name='sphere1')
sc.show(callback=scene_callback,callback_period=1./30, line_settings={'point_size':20}, start_loop=True, visible=    True, viewer='gl')
