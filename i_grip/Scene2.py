#!/usr/bin/env python3

import threading
import time
from scipy.interpolate import CubicSpline

import cv2
import numpy as np
import trimesh as tm
import pyfqmr
import pyglet

from i_grip.utils2 import *
from i_grip import HandDetectors2 as hd
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class MeshScene(tm.Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    
     
    def start(self):
        """
        Start the pyglet main loop.
        """
        pyglet.app.run()
        
    def stop(self):
        """
        Stop the pyglet main loop.
        """
        pyglet.app.exit()
        self.on_close()

class Scene :
    
    _DEFAULT_VIDEO_RENDERING_OPTIONS=dict(write_fps = True, 
                                    draw_hands=True, 
                                    draw_objects=False, 
                                    write_hands_pos = False, 
                                    write_objects_pos = False,
                                    render_objects=True)
    
    _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS = dict(is_displayed = True,
                                                    draw_grid = True, 
                                                    show_velocity_cone = True
                                                    )
    
    def __init__(self, cam_data, name = 'Grasping experiment',  video_rendering_options = _DEFAULT_VIDEO_RENDERING_OPTIONS, scene_rendering_options = _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS, fps = 30.0, detect_grasping = True, draw_mesh = True) -> None:
        self.hands = dict()
        self.objects = dict()
        self.cam_data = cam_data
        Bbox.set_image_resolution(cam_data['resolution'])
        self.rendering_options = video_rendering_options
        self.scene_rendering_options = scene_rendering_options
        self.show_velocity_cone = scene_rendering_options['show_velocity_cone']
        self.time_scene = time.time()
        self.time_hands = self.time_scene
        self.time_objects = self.time_hands
        self.time_detections = self.time_hands
        self.fps_scene = 0
        self.fps_hands= 0
        self.fps_objects= 0
        self.fps_detections= 0
        self.name = name
        self.draw_mesh = draw_mesh
        
        self.new_hand_meshes = []
        self.new_object_meshes = []
        self.hands_to_delete = {}
        self.objects_to_delete = {}
        self.scene_callback_period = 1.0/fps
        self.fps = fps
        self.scene_window = None
        self.detect_grasping = detect_grasping
        
        if self.draw_mesh:
            self.define_mesh_scene()
        self.velocity_cone_mode = 'rays'
        
            
    def __str__(self) -> str:
        s = 'SCENE \n OBJECTS :\n'
        objs = self.objects.copy().items()
        for obj in objs:
            s+=str(obj)+'\n'
        return s

    def reset(self):   
        if self.scene_window is None:
            return
        print('reset scene')
        self.stop()
        # self.hands_to_delete = self.hands
        # self.objects_to_delete = self.objects
        self.hands = dict()
        self.objects = dict()
        self.time_scene = time.time()
        self.time_hands = self.time_scene
        self.time_objects = self.time_hands
        self.fps_scene = 0
        self.fps_hands= 0
        self.fps_objects= 0
        self.new_hand_meshes = []
        self.new_object_meshes = []
        self.scene_callback_period = 1.0/self.fps
        self.resume_scene_display()
        self.define_mesh_scene()
    

    def define_mesh_scene(self):
        self.mesh_scene= tm.Scene()
        self.mesh_scene.camera.resolution = self.cam_data['resolution']
        self.mesh_scene.camera.focal= (self.cam_data['matrix'][0,0], self.cam_data['matrix'][1,1])
        self.mesh_scene.camera.z_far = 3000
        print('self.cam_data', self.cam_data)
        X =  0
        Y =  -250
        Z = 1000
        self.test_cone = tm.creation.cone(50,500)
        # self.mesh_scene.add_geometry(self.test_cone, geom_name='test_cone')
        self.mesh_scene.camera_transform = tm.transformations.rotation_matrix(np.pi, np.array([0,1,0]), np.array([0,0,0]))
        frame_origin = np.array([[X, Y, Z],
                    [X, Y, Z],[X, Y, Z]])
        frame_axis_directions = np.array([[0, 0, 1],
                        [0, 1, 0],[1, 0, 0]])
        frame_visualize = tm.load_path(np.hstack((
        frame_origin,
        frame_origin + frame_axis_directions*100)).reshape(-1, 2, 3))
        self.mesh_scene.add_geometry(frame_visualize, geom_name='mafreme')
        plane = tm.path.creation.grid(300, count = 10, plane_origin =  np.array([X, Y, Z]), plane_normal = np.array([0,1,0]))
        cam = tm.creation.camera_marker(self.mesh_scene.camera, marker_height = 300)

        self.mesh_scene.add_geometry(plane, geom_name='plane')
        
        
        self.t_scene = threading.Thread(target=self.display_meshes)
        self.run_scene_display = True
        self.t_scene.start()
        
    def display_meshes(self):
        print('Starting mesh display thread')
        self.scene_window = self.mesh_scene.show(callback=self.update_meshes,callback_period=self.scene_callback_period, line_settings={'point_size':20}, start_loop=False)
        pyglet.app.run()
        print('Mesh display thread closed')

    def pause_scene_display(self):
        print('pause scene display')
        self.run_scene_display = False
    
    def resume_scene_display(self):
        print('resume scene display')
        self.run_scene_display = True
       
    def check_all_targets(self, scene):
        targets = {}
        hands = self.hands.copy()
        objs = self.objects.copy()
        for hlabel, hand in hands.items(): 
            for olabel, obj in objs.items():
                if olabel in scene.geometry:
                    mesh = scene.geometry[olabel]
                    mesh_frame_locations = hand.check_target(obj, mesh)
                    scene.delete_geometry(hand.label+obj.label+'ray_impacts')
                    if len(mesh_frame_locations)>0:
                        impacts = tm.points.PointCloud(mesh_frame_locations, colors=hand.color)
                        # scene.add_geometry(impacts, geom_name=hand.label+'ray_impacts')
                        scene.add_geometry(impacts, geom_name=hand.label+obj.label+'ray_impacts',transform = obj.get_mesh_transform())
                else:
                    print('Mesh '+olabel+' has not been loaded yet')
            targets[hlabel]= hand.fetch_targets()
            
        for olabel in objs:
            target_info = (False, None, None)
            for hlabel in hands:
                if olabel in targets[hlabel]:
                    target_info=(True, self.hands[hlabel], targets[hlabel][olabel])
                self.objects[olabel].set_target_info(target_info)


    def evaluate_grasping_intention(self):
        hands = self.hands.copy().values
        objs = self.objects.values()
        for obj in objs():
            for hand in hands:
                if hand.label=='right':
                    obj.is_targeted_by(hand)

    def next_timestamp(self):
        for hand in self.hands.values():
            hand.next_timestamp()
        for obj in self.objects.values():
            obj.next_timestamp()

    def new_hand(self, label, input= None, timestamp = None):
        self.hands[label] = GraspingHand(label=label, hand_prediction = input, timestamp = timestamp, compute_velocity_cone=self.show_velocity_cone)
        self.new_hand_meshes.append({'mesh' : self.hands[input.label].mesh_origin, 'name': input.label})
    

    def update_hands(self, hands_predictions, timestamp = None):
        if timestamp is None:
            timestamp = time.time()
        self.update_hands_time(timestamp = timestamp)
        hands = self.hands.copy()
        for hand_pred in hands_predictions:
            if hand_pred.label not in hands:
                self.new_hand(hand_pred.label, input=hand_pred, timestamp = timestamp)
            else : 
                self.hands[hand_pred.label].update2(hand_pred)
        self.clean_hands(hands_predictions)
        self.propagate_hands( timestamp = timestamp)
        # self.evaluate_grasping_intention()
        
    def propagate_hands(self, timestamp = None):
        hands = self.hands.copy().values()
        for hand in hands:
            hand.propagate2(timestamp = timestamp)
    
    def clean_hands(self, newhands):
        hands_label = [hand.label for hand in newhands]
        hands = self.hands.copy()
        for label in hands:
            self.hands[label].setvisible(label in hands_label)
    

    def new_object(self, label, input = None, timestamp = None):
        self.objects[label] = RigidObject.from_prediction(input, timestamp)
        self.new_object_meshes.append({'mesh' : self.objects[label].mesh, 'name': label})
        
    def update_objects(self, objects_predictions:dict, timestamp = None):
        if timestamp is None:
            timestamp = time.time()
        self.update_objects_time()
        objs = self.objects.copy()
        for label, prediction in objects_predictions.items():
            if label in objs:
                self.objects[label].update(prediction, timestamp = timestamp)
            else:                    
                self.new_object(label, input=prediction, timestamp = timestamp)
        self.clean_objects()
    
    def clean_objects(self):
        todel=[]
        objs = self.objects.copy()
        for label in objs.keys():
            if self.objects[label].nb_updates <=0:
                todel.append(label)
            self.objects[label].nb_updates-=1
        for key in todel:
            del self.objects[key]
            print('object '+key+' forgotten')
     
    def update_meshes(self, scene):
        if self.run_scene_display:
            self.update_hands_meshes(scene)
            self.update_object_meshes(scene)
            if self.detect_grasping:
                self.check_all_targets(scene)

    def update_hands_meshes(self, scene):
        hands_to_delete = self.hands_to_delete.copy().keys()
        self.hands_to_delete = {}
        new_hand_meshes = self.new_hand_meshes.copy()    
        self.new_hand_meshes = []
        hands = self.hands.copy().items()
        
        if len(hands_to_delete)>0:
            print
            print(f'scene geometry {scene.geometry}')
        for label in hands_to_delete:
            scene.delete_geometry(label)
            print(f'delete hands {label}')
        if len(hands_to_delete)>0:
            print(f'scene geometry {scene.geometry}')
        
        for i in range(len(new_hand_meshes)):
            new = new_hand_meshes.pop(0)
            scene.add_geometry(new['mesh'], geom_name = new['name'])
        
        for label, hand in hands:      
            hand.update_mesh()       
            scene.graph.update(label,matrix = hand.get_mesh_transform(), geometry = label)
            if self.show_velocity_cone:
                scene.delete_geometry(hand.label+'vel_cone')
                scene.add_geometry(hand.ray_visualize, geom_name=hand.label+'vel_cone')

    def update_object_meshes(self, scene):     
        objects_to_delete = self.objects_to_delete.copy().keys()
        self.objects_to_delete = {}
        new_object_meshes = self.new_object_meshes.copy()
        self.new_object_meshes = []
        objects = self.objects.copy().items()
        
        if len(objects_to_delete)>0:
            print('avant delete')
            print(f'scene geometry {scene.geometry}')   
        for label in objects_to_delete:
            scene.delete_geometry(label)
            print(f'delete objects {label}')
        if len(objects_to_delete)>0:
            print('apres delete')
            print(f'scene geometry {scene.geometry}')
            
        for i in range(len(new_object_meshes)):
            new = new_object_meshes.pop(0)
            scene.add_geometry(new['mesh'], geom_name = new['name'])
            # self.objects_collider.add_object(new['name'], new['mesh'])            
            print('add here')
            print(scene.geometry)
            print('add there')

        for label, obj in objects:
            obj.update_mesh()
            scene.graph.update(label, matrix=obj.get_mesh_transform(), geometry = label)


    def update_hands_time(self, timestamp = None):
        if timestamp is None:
            now = time.time()
        else:
            now = timestamp
        self.elapsed_hands= now - self.time_hands 
        hands = self.hands.copy().values()
        for hand in hands:
            hand.set_elapsed(self.elapsed_hands)
        self.fps_hands = 1 / self.elapsed_hands
        self.time_hands = now

    def update_scene_time(self, timestamp = None):
        if timestamp is None:
            now = time.time()
        else:
            now = timestamp
        self.elapsed_scene= now - self.time_scene
        self.fps_scene = 1 / self.elapsed_scene
        self.time_scene = now

    def update_objects_time(self, timestamp = None):
        if timestamp is None:
            now = time.time()
        else:
            now = timestamp
        self.elapsed_objects = now - self.time_objects
        self.fps_objects = 1 /self.elapsed_objects
        self.time_objects = now
    
    def update_detections_time(self, timestamp = None):
        if timestamp is None:
            now = time.time()
        else:
            now = timestamp
        self.elapsed_detections = now - self.time_detections
        self.fps_detections = 1 /self.elapsed_detections
        self.time_detections = now


    def compute_distances(self):
        hands = self.hands.copy().values()
        objs = self.objects.copy().values()
        for obj in objs:
            for hand in hands:
                obj.distance_to(hand)

    def write_fps(self, img):
        cv2.putText(img,'fps scene: {:.0f}'.format(self.fps_scene),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(img,'fps hands: {:.0f}'.format(self.fps_hands),(10,65),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(img,'fps detections: {:.0f}'.format(self.fps_detections),(10,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(img,'fps objects: {:.0f}'.format(self.fps_objects),(10,115),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    def stop(self):
        #stop the callback thread
        print('stopped pyglet app inside scene')
        self.scene_window.on_close()
        print('stopping inside scene')
        pyglet.app.exit()
        print('stopped scene window inside scene')
        self.t_scene.join()
        print('joined scene thread inside scene')
        

    def get_hands_data(self):
        data = {}
        # loop over hands
        
        for hand in self.hands.values():
            data[hand.label + '_hand'] = hand.get_trajectory()
            print(f"get_{hand.label}_data: {data[hand.label + '_hand_traj']}")
        return data
    
    def get_objects_data(self):
        data = {}
        for obj in self.objects.values():
            data[obj.label] = obj.get_trajectory()
            print(f"get_{obj.label}_data: {data[obj.label + '_obj_traj']}")
        return data
    

class LiveScene(Scene):
    _DEFAULT_VIDEO_RENDERING_OPTIONS=dict(write_fps = True, 
                                    draw_hands=True, 
                                    draw_objects=False, 
                                    write_hands_pos = False, 
                                    write_objects_pos = False,
                                    render_objects=True)
    
    _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS = dict(is_displayed = True,
                                                    draw_grid = True,
                                                    show_velocity_cone = True)
        
    def __init__(self, cam_data, name='Grasping experiment',   video_rendering_options = _DEFAULT_VIDEO_RENDERING_OPTIONS, scene_rendering_options = _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS) -> None:
        super().__init__(cam_data, name, video_rendering_options, scene_rendering_options)
    
    def render(self, img):
        # self.compute_distances()
        self.update_scene_time()
        hands = self.hands.copy().values()
        objs = self.objects.copy().values()
        for hand in hands:
            hand.render(img)
        for obj in objs:
            obj.render(img)
        if self.rendering_options['write_fps']:
            self.write_fps(img)

        # cv2.imshow(self.name, img)

        return img
    
class ReplayScene(Scene):
        
    _DEFAULT_VIDEO_RENDERING_OPTIONS=dict(write_fps = True, 
                                    draw_hands=True, 
                                    draw_objects=False, 
                                    write_hands_pos = False, 
                                    write_objects_pos = False,
                                    render_objects=True)
    
    _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS = dict(is_displayed = True,
                                                    draw_grid = True,
                                                    show_velocity_cone = True,
                                                    fps = 30)
    
    def __init__(self, cam_data, name='Grasping experiment',   video_rendering_options = _DEFAULT_VIDEO_RENDERING_OPTIONS, scene_rendering_options = _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS) -> None:
        super().__init__(cam_data, name, video_rendering_options, scene_rendering_options, detect_grasping=False)
    
    def render(self, img):  
        # self.compute_distances()
        self.update_scene_time()
        hands = self.hands.copy().values()
        objs = self.objects.copy().values()
        for hand in hands:
            hand.render(img)
        for obj in objs:
            obj.render(img)
        if self.rendering_options['write_fps']:
            self.write_fps(img)        
        return img


class AnalysiScene(Scene):
    _DEFAULT_VIDEO_RENDERING_OPTIONS=dict(write_fps = True, 
                                    draw_hands=True, 
                                    draw_objects=False, 
                                    write_hands_pos = False, 
                                    write_objects_pos = False,
                                    render_objects=True)
    
    _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS = dict(is_displayed = True,
                                                    draw_grid = True,
                                                    show_velocity_cone = True)
    
    def __init__(self, cam_data, name='Grasping experiment',   video_rendering_options = _DEFAULT_VIDEO_RENDERING_OPTIONS, scene_rendering_options = _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS) -> None:
        super().__init__(cam_data, name, video_rendering_options, scene_rendering_options)

    def create_void_hands(self):
        labels = ('left', 'right')
        for label in labels:
            self.hands[label] = GraspingHand(label=label, compute_velocity_cone=self.show_velocity_cone)
            self.new_hand_meshes.append({'mesh' : self.hands[label].mesh_origin, 'name': label})

    def load_hands(self, hands_predictions, timestamp = None):
        for label, hand in self.hands.items():
            if label in hands_predictions:
                hand.minimal_update(hands_predictions[label])
            else:
                hand.minimal_update(None)
        self.propagate_hands( timestamp = timestamp)
        

class Entity:
    
    MAIN_DATA_KEYS = [ 'x', 'y', 'z']
    
    def __init__(self) -> None:
        self.visible = True
        self.lost = False
        self.invisible_time = 0
        self.max_invisible_time = 0.1
        self.raw = {}
        self.refined = {}
        self.derivated = {}
        self.derivated_refined = {}
        self.refinable_keys = {}
        self.filters={}
        self.derivated_filters={}
        self.elapsed=0
        self.velocity = 0
        self.normed_velocity = self.velocity
        self.new= True
        self.trajectory = pd.DataFrame(columns = ['Timestamps']+Entity.MAIN_DATA_KEYS)
        self.state = None

    def setvisible(self, bool):
        if not bool:
            self.invisible_time += self.elapsed
        elif self.visible:
            self.invisible_time = 0
        self.visible = bool
        self.lost = not (self.invisible_time < self.max_invisible_time)

    def set_elapsed(self, elapsed):
        self.elapsed = elapsed

    def pose(self):
        return self.pose
    
    def position(self):
        return self.pose.position.v
    
    def velocity(self):

        return self.velocity
    
    def update_mesh(self):
        pass
    
    def update_trajectory(self):
        # add a new line to the trajectory dataframe with the current timestamp position and velocity
        self.trajectory.add(self.state)
    
    def get_trajectory(self):
        return self.trajectory.get_data()
    
    def load_trajectory(self, trajectory):
        self.trajectory = trajectory
        
    def get_mesh_transform(self):
        return self.mesh_transform
    
    def get_inv_mesh_transform(self):
        return self.inv_mesh_transform
    
class RigidObject(Entity):
    
    MAIN_DATA_KEYS=RigidObjectTrajectory.DEFAULT_DATA_KEYS
    
    _TLESS_MESH_PATH = '/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/tless/models_cad'
    _YCVB_MESH_PATH = '/home/emoullet/Documents/DATA/cosypose/local_data/bop_datasets/ycbv/models'
    _TLESS_URDF_PATH = '/home/emoullet/Documents/DATA/cosypose/local_data/urdfs/tless.cad/'
    _YCVB_URDF_PATH = '/home/emoullet/Documents/DATA/cosypose/local_data/urdfs/ycbv/'
    
    def __init__(self, dataset = 'tless',  label = None, pose=None, score = None, render_box=None, timestamp = None, trajectory = None) -> None:
        super().__init__()
        if(dataset == "ycbv"):
            self.mesh_path = RigidObject._YCVB_MESH_PATH
            self.urdf_path = RigidObject._YCVB_URDF_PATH
        elif(dataset == "tless"):
            self.mesh_path = RigidObject._TLESS_MESH_PATH
            self.urdf_path = RigidObject._TLESS_URDF_PATH
            
        self.label = label
        self.default_color = (0, 255, 0)
        
        if pose is None or timestamp is None:
            self.state = None
            if pose is not None and timestamp is None:
                raise ValueError('timestamp must be provided if pose is provided')
        else:
            self.state = RigidObjectState.from_prediction(pose, timestamp)
        self.trajectory = RigidObjectTrajectory.from_state(self.state)
        #TODO : add traj to init
        self.score = score
        if render_box is None:
            self.render_box = None
        else:
            self.render_box = Bbox(label, render_box)
        self.distances={}
        print('object '+self.label+ ' discovered')
        self.nb_updates = 10
        self.target_metric = 0
        
        self.appearing_radius = 0.2
        self.simplify = False
        self.load_simplified = True
        self.load_mesh()
        # self.load_urdf()
        self.mesh_pos = np.array([0,0,0])
        self.mesh_transform = np.identity(4)
        
        self.is_targeted = False
        self.targeter = None
        self.target_info = None
        self.update_display()
        
        # self.trajectory = pd.DataFrame(columns=['Timestamps', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'is_targeted', 'targeter', 'time_of_impact', 'grip'])
    
    @classmethod
    def from_prediction(cls, prediction, timestamp):
        return cls(dataset=prediction['dataset'], label = prediction['label'], pose = prediction['pose'], score = prediction['score'], render_box = prediction['render_box'], timestamp = timestamp)
    @classmethod
    def from_trajectory(cls, label, trajectory):
        return cls(label = label, trajectory)
    
    def __str__(self):
        out = 'label: ' + str(self.label) + '\n pose: {' +str(self.pose)+'} \n nb_updates: '+str(self.nb_updates)
        return out
    
    def update(self, new_prediction, timestamp = None):
        self.state.update(new_prediction['pose'], timestamp = timestamp)
        self.render_box.update_coordinates(new_prediction['render_box'])
        if self.nb_updates <=15:
            self.nb_updates+=2
        self.update_trajectory()
    
    def update_pose(self, translation_vector, quaternion):
        self.pose.update_from_vector_and_quat(translation_vector, quaternion)

    def update_from_trajectory(self, row_index):
        # read line from trajectory
        row = self.trajectory.iloc[row_index]
        timestamp = row['Timestamps']
        xyz = row[['x', 'y', 'z']].values
        orient = row[['qx', 'qy', 'qz', 'qw']].values
        self.state.update_from_vector_and_quat(xyz, orient, timestamp = timestamp)
        
    def load_mesh(self):
        try :
            if self.load_simplified:
                self.mesh = tm.load_mesh(self.mesh_path+'_simplified/'+self.label+'.ply')
                print('MESH LOADED : ' + self.mesh_path+'_simplified/'+self.label+'.ply')
            else:
                self.mesh = tm.load_mesh(self.mesh_path+'/'+self.label+'.ply')
                print('MESH LOADED : ' + self.mesh_path+'/'+self.label+'.ply')
            if self.simplify:
                mesh_simplifier = pyfqmr.Simplify()
                mesh_simplifier.setMesh(self.mesh.vertices,self.mesh.faces)
                mesh_simplifier.simplify_mesh(target_count = 1000, aggressiveness=7, preserve_border=True, verbose=10)
                v, f, n = mesh_simplifier.getMesh()
                self.mesh = tm.Trimesh(vertices=v, faces=f, face_normals=n)
            print(len(self.mesh.vertices))
        except:
            self.mesh = None
            print(self.mesh_path)
            print('MESH LOADING FAILED')
    
    def load_urdf(self):
        self.mesh = tm.load_mesh(self.urdf_path+self.label+'/'+self.label+'.obj')
        print(len(self.mesh.vertices))
        print('URDF LOADED : ' + self.urdf_path+self.label+'/'+self.label+'.obj')
        exit()

    def update_mesh(self):
        self.mesh_pos = self.state.pose_filtered.position.v*np.array([-1,1,1])
        # mesh_orient_quat = [self.pose.orientation.q[i] for i in range(4)]
        # mesh_orient_angles = self.pose.orientation.v*np.array([-1,-1,-1])+np.pi*np.array([1  ,1,0])
        mesh_orient_angles = self.state.pose_filtered.orientation.v*np.array([1,1,1])+np.pi*np.array([0 ,0,1])
        # x_reflection_matrix = tm.transformations.reflection_matrix(np.array([0,0,0]), np.array([1,0,0]))
        #mesh_transform = tm.transformations.translation_matrix(mesh_pos)  @ tm.transformations.quaternion_matrix(mesh_orient_quat)
        rot_mat = tm.transformations.euler_matrix(mesh_orient_angles[0],mesh_orient_angles[1],mesh_orient_angles[2])
        self.mesh_transform = tm.transformations.translation_matrix(self.mesh_pos) @ rot_mat
        self.inv_mesh_transform = np.linalg.inv(self.mesh_transform)
    
    def write(self, img):
        text = self.label 
        x = self.render_box.corner1[0]
        y = self.render_box.corner1[1]-60
        dy = 15
        cv2.rectangle(img, (x,y-20), (self.render_box.corner2[0],self.render_box.corner1[1]), (200,200,200), -1)
        cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color)    
        if self.is_targeted:
            text ='Trgt by : ' + self.targeter.label 
            y+=dy
            cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color)  
            text = 'tbi : '+str(self.target_info.get_time_of_impact()) + 'ms'
            y+=dy
            cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color)  
            text ='GRIP : '+self.target_info.get_grip()
            y+=dy
            cv2.putText(img, text , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color)  

    def write_dist(self, img):
        text = self.label
        x = self.render_box.corner1[0]
        y = self.render_box.corner2[1]
        dy = 20
        cv2.rectangle(img, (x,y), (self.render_box.corner2[0]+30,y+50), (200,200,200), -1)
        for k, d in self.distances.items():
            cv2.putText(img, 'd-'+k+' : '+str(int(d)) +' cm' , (x,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.color)    
            y+=dy

    def render(self, img, bbox = True, txt=True, dist=False, overlay=False):
        self.update_display()
        if txt:
            self.write(img)
        if bbox:
            self.render_box.draw(img)
        if dist:
            self.write_dist(img)
        if overlay:
            pass
    
    # def distance_to(self, hand):
    #     self.distances[hand.label] = np.linalg.norm(100*self.pose.position.v - 0.1*hand.xyz)


    def set_target_info(self, info):
        self.is_targeted = info[0]
        self.targeter = info[1]
        self.target_info = info[2]
        #if self.is_targeted:
        #    print(self.label, 'is targeted by ', self.targeter.label, 'in', self.target_info.get_time_of_impact())

    def update_display(self):
        if self.is_targeted:
            self.color = self.targeter.text_color
            thickness = 4
        else:
            self.color = self.default_color
            thickness = 2
        self.render_box.update_display(self.color, thickness)
        # print('DISPLAY UPDATED', time.time())

    
class GraspingHand(Entity):
    MAIN_DATA_KEYS=GraspingHandTrajectory.DEFAULT_DATA_KEYS
    
    def __init__(self, label = None, hand_prediction=None, timestamp=None, trajectory = None, compute_velocity_cone = False)-> None:
        super().__init__()
        self.label = label
        if trajectory is None:
            self.detected_hand = hand_prediction
            if isinstance(hand_prediction, hd.HandDetectors.HandPrediction):
                self.state = GraspingHandState.from_hand_detection(hand_prediction, timestamp = timestamp)
                if self.label is None:
                    self.label = hand_prediction.label
            elif isinstance(hand_prediction, np.ndarray):
                self.state = GraspingHandState.from_position(hand_prediction, timestamp = timestamp)
            else:   
                self.state = None
            
            # define trajectory as a pandas dataframe with columns : time, x, y, z, vx, vy, vz, v
            self.trajectory = GraspingHandTrajectory.from_state(self.state)
        elif isinstance(trajectory, pd.DataFrame):
            self.trajectory = GraspingHandTrajectory.from_dataframe(trajectory)
            self.state = GraspingHandState.from_position(self.trajectory.next_position())
        
        self.new= True
        self.visible = True
        self.invisible_time = 0
        self.max_invisible_time = 0.3
 
        # self.update_trajectory(timestamp)
        self.compute_velocity_cone = compute_velocity_cone
        self.show_label = True
        self.show_xyz = True
        self.show_roi = True
        self.margin = 10  # pixels
        self.font_size = 1
        self.font_thickness = 1
        self.handedness_text_color = (88, 205, 54) # vibrant green
        self.font_size_xyz = 0.5
    
        ### define mesh
        self.define_mesh_representation()
        self.define_velocity_cone()
        self.update_mesh()
        
        self.target_detector = TargetDetector(self.label)

    @classmethod
    def from_trajectory(cls, trajectory):
        return cls(trajectory = trajectory)
    
    def build_from_hand_prediction(self, hand_prediction):
        self.__dict__= hand_prediction.__dict__
        if isinstance(hand_prediction, i_grip.HandDetectors.HandPrediction):
            self.draw = hand_prediction.draw
        self.base_dict = hand_prediction.__dict__
        
    def build_from_scratch(self, label):        
        self.label = label
        self.state = GraspingHandState()
        
    def define_mesh_representation(self):
        self.mesh_origin = tm.primitives.Sphere(radius = 20)
        if self.label == 'right':
            self.mesh_origin.visual.face_colors = self.color = [255,0,0,255]
            self.text_color = [0,0,255]
        else:  
            self.mesh_origin.visual.face_colors = self.color = [0,0,100,255]
            self.text_color=[100,0,0]
    
    def define_velocity_cone(self,cone_max_length = 500, cone_min_length = 50, vmin=10, vmax=200, cone_max_diam = 100,cone_min_diam = 20, n_layers=5):
        self.cone_angle = np.pi/8
        self.n_layers = n_layers
        self.cone_lenghts_spline = spline(vmin,vmax, cone_min_length, cone_max_length)
        self.cone_diam_spline = spline(vmin,vmax, cone_min_diam, cone_max_diam)
        #self.cone_diam_spline = spline(vmin,vmax, cone_max_diam, cone_min_diam)
    
    def update2(self, detected_hand):
        self.detected_hand = detected_hand
        self.state.update(detected_hand)
    
    # def minimal_update(self, pos):
    #     self.xyz = pos*1000
    #     self.raw['xyz'] = pos*1000
    
    def update_mesh(self):
        self.mesh_position = Position(self.state.position_filtered*np.array([-1,1,1]))
        # self.mesh_position = Position(self.state.velocity_filtered*np.array([-1,1,1])+np.array([0,0,500]))
        self.mesh_transform= tm.transformations.compose_matrix(translate = self.mesh_position.v)
        if self.compute_velocity_cone:
            self.make_rays()
        
        
    def update_from_trajectory(self, row_index):
        row = self.trajectory.iloc[row_index]
        timestamp = row['Timestamps']
        pos = row[['x', 'y', 'z']].values
        J
        
    def propagate2(self, timestamp=None):
        self.state.propagate(timestamp)
        self.update_trajectory()
        # self.update_meshes()
              


    def make_rays(self): 
        vdir = self.state.normed_velocity
        svel = self.state.scalar_velocity
        if True:
        # if svel > 10:
            cone_len = self.cone_lenghts_spline(svel)
            cone_diam = self.cone_diam_spline(svel)
            # print('vdir', vdir)
            # print('svel', svel)
            # print('cone_len', cone_len)
            # print('cone_diam', cone_diam)
            # print('self.mesh_position.v', self.mesh_position.v)
            # vecteur aléatoire
            random_vector = np.random.rand(len(vdir))

            # projection
            projection = np.dot(random_vector, vdir)

            # vecteur orthogonal
            orthogonal_vector = random_vector - projection * vdir

            # vecteur unitaire orthogonal
            orthogonal_unit_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
            orthogonal_unit_vector2 = np.cross(vdir, orthogonal_unit_vector)

            self.ray_directions_list = [ vdir*cone_len + (-cone_diam/2+i*cone_diam/(self.n_layers-1)) * orthogonal_unit_vector+ (-cone_diam/2+j*cone_diam/(self.n_layers-1)) * orthogonal_unit_vector2 for i in range(self.n_layers) for j in range(self.n_layers) ] 

            self.ray_origins = np.vstack([self.mesh_position.v for i in range(self.n_layers) for j in range(self.n_layers) ])
            self.ray_directions = np.vstack(self.ray_directions_list)
            self.ray_visualize = tm.load_path(np.hstack((
                self.ray_origins,
                self.ray_origins + self.ray_directions)).reshape(-1, 2, 3))     
        else:
            self.ray_visualize = None

    def check_target(self, obj, mesh):
        inv_trans = obj.inv_mesh_transform
        hand_pos_obj_frame = (inv_trans@self.mesh_position.ve)[:3]
        # ray_directions_list = [ np.array([0,-1000,0]) + np.array([-200+i*100,0,-200+j*100] ) for i in range(5) for j in range(5) ]
        rot = inv_trans[:3,:3]
        ray_origins_obj_frame = np.vstack([ hand_pos_obj_frame for i in range(self.n_layers) for j in range(self.n_layers) ])
        ray_directions_obj_frame = np.vstack([rot @ ray_dir for ray_dir in self.ray_directions_list])    
        try:
            mesh_frame_locations, _, _ = mesh.ray.intersects_location(ray_origins=ray_origins_obj_frame,
                                                                            ray_directions=ray_directions_obj_frame,
                                                                            multiple_hits=False)
        except:
            mesh_frame_locations = ()
        self.target_detector.new_impacts(obj, mesh_frame_locations, hand_pos_obj_frame, self.elapsed)
        return mesh_frame_locations


    def fetch_targets(self):
        self.most_probable_target, targets = self.target_detector.get_most_probable_target()
        return targets
    
    def render(self, img):
        # Draw the hand landmarks.
        # displayed_landmarks = self.state.landmarks
        displayed_landmarks = self.detected_hand.get_landmarks()
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        # landmarks_proto.landmark.extend([
        #                                     landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in displayed_landmarks])
        landmarks_proto.landmark.extend([
                                            landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in displayed_landmarks])
        solutions.drawing_utils.draw_landmarks(
                                            img,
                                            landmarks_proto,
                                            solutions.hands.HAND_CONNECTIONS,
                                            solutions.drawing_styles.get_default_hand_landmarks_style(),
                                            solutions.drawing_styles.get_default_hand_connections_style())

        if self.show_label:
            # Get the top left corner of the detected hand's bounding box.
            text_x = int(min(displayed_landmarks[:,0]))
            text_y = int(min(displayed_landmarks[:,1])) - self.margin

            # Draw handedness (left or right hand) on the image.
            # cv2.putText(img, f"{self.handedness[0].category_name}",
            #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
            #             self.font_size, self.handedness_text_color, self.font_thickness, cv2.LINE_AA)
            
            cv2.putText(img, f"{self.label}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        self.font_size, self.handedness_text_color, self.font_thickness, cv2.LINE_AA)
        if self.show_roi:
            cv2.rectangle(img, (self.detected_hand.roi[0],self.detected_hand.roi[1]),(self.detected_hand.roi[2],self.detected_hand.roi[3]),self.handedness_text_color)

        if self.show_xyz:
            # Get the top left corner of the detected hand's bounding box.z
            
            #print(f"{self.label} --- X: {self.xyz[0]/10:3.0f}cm, Y: {self.xyz[0]/10:3.0f} cm, Z: {self.xyz[0]/10:3.0f} cm")
            if len(img.shape)<3:
                height, width = img.shape
            else:
                height, width, _ = img.shape
            x_coordinates = displayed_landmarks[:,0]
            y_coordinates = displayed_landmarks[:,1]
            x0 = int(max(x_coordinates) * width)
            y0 = int(max(y_coordinates) * height) + self.margin

            # Draw handedness (left or right hand) on the image.
            cv2.putText(img, f"X:{self.state.position_filtered.x/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_DUPLEX, self.font_size_xyz, (20,180,0), self.font_thickness, cv2.LINE_AA)
            cv2.putText(img, f"Y:{self.state.position_filtered.y/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_DUPLEX, self.font_size_xyz, (255,0,0), self.font_thickness, cv2.LINE_AA)
            cv2.putText(img, f"Z:{self.state.position_filtered.z/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_DUPLEX, self.font_size_xyz, (0,0,255), self.font_thickness, cv2.LINE_AA)

        
def rotation_from_vectors(v1, v2):

    # Calcul de l'axe et de l'angle de rotation
    axis = np.cross(v1, v2)
    if np.linalg.norm(v1) * np.linalg.norm(v2) == 0:
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    else:
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # # Création de la rotation à partir de l'axe et de l'angle
    # r = R.from_rotvec(angle * axis)

    # # Obtention du quaternion correspondant à la rotation
    # q = r.as_quat()
    q = tm.transformations.rotation_matrix(angle, axis)
    return q

def spline(x0,x1, y0, y1):

    x=np.array([x0,x1])
    y=np.array([y0, y1])
    cs = CubicSpline(x,y, bc_type='clamped')
    def eval(x):
        if x<=x0:
            return y0
        elif x>=x1:
            return y1
        else:
            return cs(x)
    return eval