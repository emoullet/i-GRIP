#!/usr/bin/env python3

import threading
import time

import cv2
import numpy as np
import trimesh as tm
import pyglet


from i_grip.utils2 import *   
# from i_grip.Hands import GraspingHand
from i_grip.Hands_refactored import GraspingHand
from i_grip.Objects import RigidObject
# from i_grip.Targets_refactored_multi_and_fullthread import TargetDetector
# from i_grip.Targets_refactored_fullmulti import TargetDetector
from i_grip.Targets_refactored_fullmulti_multichecker import TargetDetector
# from i_grip.Targets_refactored_multi import TargetDetector
from i_grip.clean_scene import CleanScene

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
    
    def __init__(self, cam_data, name = 'Grasping experiment',  video_rendering_options = _DEFAULT_VIDEO_RENDERING_OPTIONS, scene_rendering_options = _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS, fps = 30.0, detect_grasping = True, draw_mesh = True, dataset = None, plotter=None) -> None:
        self.hands = dict()
        self.objects = dict()
        self.target_detectors = dict()
        self.cam_data = cam_data
        Bbox.set_image_resolution(cam_data['resolution'])
        self.rendering_options = video_rendering_options
        self.scene_rendering_options = scene_rendering_options
        self.show_velocity_cone = scene_rendering_options['show_velocity_cone']
        self.show_trajectory = True
        self.show_prediction = True
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
        self.dataset= dataset
        self.plotter = plotter
        
        self.timestep_index = 0
        
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
        self.target_detectors = dict()
        self.time_scene = time.time()
        self.time_hands = self.time_scene
        self.time_objects = self.time_hands
        self.fps_scene = 0
        self.fps_hands= 0
        self.fps_objects= 0
        self.new_hand_meshes = []
        self.new_object_meshes = []
        self.scene_callback_period = 1.0/self.fps
        self.timestep_index = 0
        self.resume_scene_display()
        self.define_mesh_scene()
    

    def define_mesh_scene(self):
        self.mesh_scene= CleanScene()
        # self.mesh_scene= tm.Scene()
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
        self.scene_window = self.mesh_scene.show(callback=self.update_meshes,callback_period=self.scene_callback_period, line_settings={'point_size':20}, start_loop=False, visible=    True, viewer='gl')
        # time.sleep(0.5)
        pyglet.app.run()
        print('Mesh display thread closed')

    def pause_scene_display(self):
        print('pause scene display')
        self.run_scene_display = False
    
    def resume_scene_display(self):
        print('resume scene display')
        self.run_scene_display = True
       
    def next_timestamp(self, timestamp):
        for hand in self.hands.values():
            hand.update_from_trajectory()
        for obj in self.objects.values():
            obj.update_from_trajectory()
        self.timestep_index +=1
        print('timestep index : '+str(self.timestep_index))
        self.fetch_all_targets(timestamp=timestamp)

    def new_hand(self, label, input= None, timestamp = None):
        new_hand = GraspingHand(label=label, input = input, timestamp = timestamp, plotter=self.plotter)
        self.hands[label] = new_hand
        self.new_hand_meshes.append({'mesh' : self.hands[label].mesh_origin, 'name': label})
        if new_hand.full_hand:
            for i, key_point in enumerate(new_hand.mesh_key_points):
                self.new_hand_meshes.append({'mesh' : key_point, 'name': label+'_keypoint_'+str(i)})
        
        self.target_detectors[label] = TargetDetector(new_hand, plotter= self.plotter)
        
        obj_labels = self.objects.keys()
        for obj_label in obj_labels:
            self.target_detectors[label].new_target(self.objects[obj_label])
    
    def update_hands(self, hands_predictions, timestamp = None):
        if timestamp is None:
            timestamp = time.time()
        self.update_hands_time(timestamp = timestamp)
        hands = self.hands.keys()
        for hand_pred in hands_predictions:
            # if hand_pred.label == 'left':
            #     return
            if hand_pred.label not in hands:
                self.new_hand(hand_pred.label, input=hand_pred, timestamp = timestamp)
            else : 
                self.hands[hand_pred.label].update(hand_pred)
        self.clean_hands(hands_predictions)
        self.propagate_hands( timestamp = timestamp)
        
        # keys = self.target_detectors.copy().keys()
        # for label in keys:
        #     self.target_detectors[label].update_hand(self.hands[label])
        # self.evaluate_grasping_intention()
        
    def propagate_hands(self, timestamp = None):
        keys = self.hands.keys()
        for hand_label in keys:
            self.hands[hand_label].propagate(timestamp = timestamp)
    
    def clean_hands(self, newhands):
        hands_label = [hand.label for hand in newhands]
        hands = self.hands.copy()
        for label in hands:
            self.hands[label].setvisible(label in hands_label)
    

    def new_object(self, label, input = None, timestamp = None, dataset = None):
        self.objects[label] = RigidObject(input, timestamp = timestamp, dataset = dataset, label = label, index= len(self.objects))
        
        print('new object '+label)
        
        target_detector_labels = self.target_detectors.copy().keys()
        for target_label in target_detector_labels:
            self.target_detectors[target_label].new_target(self.objects[label])
        self.new_object_meshes.append({'mesh' : self.objects[label].mesh, 'name': label})
        
    def update_objects(self, objects_predictions:dict, timestamp = None):
        if timestamp is None:
            timestamp = time.time()
        self.update_objects_time()
        obj_labels = self.objects.keys()
        for label, prediction in objects_predictions.items():
            if label in obj_labels:
                self.objects[label].update(prediction, timestamp = timestamp)
            else:                    
                self.new_object(label, input=prediction, timestamp = timestamp, dataset = self.dataset)
        self.clean_objects()
        
        # detector_labels = self.target_detectors.copy().keys()
        # for label in detector_labels:
        #     self.target_detectors[label].update_objects(self.objects.values())
    
    def clean_objects(self):
        todel=[]
        objs = self.objects.keys()
        for label in objs:
            if self.objects[label].nb_updates <=0:
                todel.append(label)
            self.objects[label].nb_updates-=1
        for key in todel:
            del self.objects[key]
            print('object '+key+' forgotten')
     
    def update_meshes(self, scene):
        ttot = time.time()
        if self.run_scene_display:
            # print('update meshes')
            t = time.time()
            if self.show_prediction:
                self.predict_future_trajectory(scene)
            print(f'predict_future_trajectory time : {(time.time()-t)*1000:.2f} ms')
            t = time.time()
            self.update_hands_meshes(scene)
            print(f'update_hands_meshes time : {(time.time()-t)*1000:.2f} ms')
            t = time.time()
            self.update_object_meshes(scene)
            print(f'update_object_meshes time : {(time.time()-t)*1000:.2f} ms')
            t= time.time()
            self.update_target_detectors_meshes(scene)
            print(f'update_target_detectors_meshes time : {(time.time()-t)*1000:.2f} ms')
            t = time.time()
            if self.show_trajectory:
                self.update_trajectory_meshes(scene)
            print(f'update_trajectory_meshes time : {(time.time()-t)*1000:.2f} ms')
            t = time.time()
            if self.detect_grasping:
                self.check_all_targets(scene)
                print(f'check_all_targets time : {(time.time()-t)*1000:.2f} ms')
                t = time.time()
                self.fetch_all_targets()
                print(f'fetch_all_targets time : {(time.time()-t)*1000:.2f} ms')
        print(f'update_meshes time : {(time.time()-ttot)*1000:.2f} ms')

    # def update_meshes(self, scene):
    #     if self.run_scene_display:
    #         # print('update meshes')
    #         if self.show_prediction:
    #             self.predict_future_trajectory(scene)
    #         self.update_hands_meshes(scene)
    #         self.update_object_meshes(scene)
    #         self.update_target_detectors_meshes(scene)
    #         if self.show_trajectory:
    #             self.update_trajectory_meshes(scene)
    #         if self.detect_grasping:
    #             self.check_all_targets(scene)
    #             self.fetch_all_targets()
                
    def update_hands_meshes(self, scene):
        hands_to_delete = self.hands_to_delete.copy().keys()
        self.hands_to_delete = {}
        new_hand_meshes = self.new_hand_meshes.copy()    
        self.new_hand_meshes = []
        hands = self.hands.copy().items()
        
        for label in hands_to_delete:
            scene.delete_geometry(label)
        for i in range(len(new_hand_meshes)):
            new = new_hand_meshes.pop(0)
            scene.add_geometry(new['mesh'], geom_name = new['name'])
        
        for label, hand in hands:      
            t = time.time()
            hand.update_mesh()       
            print(f'update_mesh time for hand {label} : {(time.time()-t)*1000:.2f} ms')
            t = time.time()
            scene.graph.update(label,matrix = hand.get_mesh_transform(), geometry = label)
            if hand.full_hand:
                tfs, paths = hand.get_keypoints_representation()
                for i  in range(len(hand.mesh_key_points)):
                    scene.graph.update(label+'_keypoint_'+str(i), matrix = tfs[i], geometry = label+'_keypoint_'+str(i))
                scene.delete_geometry(label+'_keypoint_connection_paths')
                scene.add_geometry(paths, geom_name=label+'_keypoint_connection_paths')
            print(f'update_graph time for hand {label} : {(time.time()-t)*1000:.2f} ms')

    def update_object_meshes(self, scene):     
        objects_to_delete = self.objects_to_delete.copy().keys()
        self.objects_to_delete = {}
        new_object_meshes = self.new_object_meshes.copy()
        self.new_object_meshes = []
        objects = self.objects.copy().items()
        
        for label in objects_to_delete:
            scene.delete_geometry(label)
            
        for i in range(len(new_object_meshes)):
            new = new_object_meshes.pop(0)
            scene.delete_geometry(new['name'])
            scene.add_geometry(new['mesh'], geom_name = new['name'])
            # self.objects_collider.add_object(new['name'], new['mesh'])      

        for label, obj in objects:
            obj.update_mesh()
            scene.graph.update(label, matrix=obj.get_mesh_transform(), geometry = label)

    def update_target_detectors_meshes(self, scene):
        if self.show_velocity_cone:
            target_detector_labels = self.target_detectors.copy().keys()
            for label in target_detector_labels:
                detector = self.target_detectors[label]    
                scene.delete_geometry(detector.label+'_cone')
                scene.add_geometry(detector.get_rays(), geom_name=detector.label+'_cone')

    
    def update_trajectory_meshes(self, scene):
        t = time.time()
        hands = self.hands.copy()
        print(f'copy hands time : {(time.time()-t)*1000:.2f} ms')
        for hlabel, hand in hands.items():
            if self.show_trajectory:
                scene.delete_geometry(hand.label+'trajectory')
                scene.delete_geometry(hand.label+'extrapolated_trajectory')
                _observed_trajectory, extrapolated_trajectory = hand.get_trajectory_points()
                if len(_observed_trajectory) :
                    _observed_trajectory = tm.points.PointCloud(_observed_trajectory, colors=hand.color)
                    scene.add_geometry(_observed_trajectory, geom_name=hand.label+'trajectory')
                if len(extrapolated_trajectory):
                    extrapolated_trajectory = tm.points.PointCloud(extrapolated_trajectory, colors=hand.extrapolated_trajectory_color)
                    scene.add_geometry(extrapolated_trajectory, geom_name=hand.label+'extrapolated_trajectory')
    
    def predict_future_trajectory(self, scene):
        hands= self.hands.copy()
        for hlabel, hand in hands.items():
            t= time.time()
            scene.delete_geometry(hand.label+'future_trajectory')
            print(f'delete_geometry time for hand {hand.label} : {(time.time()-t)*1000:.2f} ms')
            t = time.time()
            predicted_trajectory = hand.get_future_trajectory_points()
            print(f'get_future_trajectory time for hand {hand.label} : {(time.time()-t)*1000:.2f} ms')
            t = time.time()
            if len(predicted_trajectory)>0:
                predicted_trajectory = tm.points.PointCloud(predicted_trajectory, colors=hand.future_color)
                scene.add_geometry(predicted_trajectory, geom_name=hand.label+'future_trajectory')
            print(f'scene geometries: {len(scene.geometry)}')
            print(f'add_geometry time for hand {hand.label} : {(time.time()-t)*1000:.2f} ms')


    def check_all_targets(self, scene, timestamp = None):
        if timestamp is None:
            timestamp = time.time()
        tall = time.time()
        target_detector_labels = self.target_detectors.copy().keys()
        tt = {}
        for label in target_detector_labels:
            tt[label] = time.time()
            self.target_detectors[label].check_all_targets()
        for label in target_detector_labels:
            scene.delete_geometry(label+'ray_impacts')
            impacts = self.target_detectors[label].get_impacts()
            if impacts is not None:
                impacts_geom = tm.points.PointCloud(impacts, colors=self.hands[label].color)
                scene.add_geometry(impacts_geom, geom_name=label+'ray_impacts')
            print(f'check_all_targets time for hand {label} : {(time.time()-tt[label])*1000:.2f} ms')
        print(f'check_all_targets time : {(time.time()-tall)*1000:.2f} ms')

    def fetch_all_targets(self, timestamp = None):
        targets = {}
        target_detector_labels = self.target_detectors.copy().keys()
        if timestamp is None:
            timestamp = time.time()
        for label in target_detector_labels:
            t = time.time()            
            targets[label], _ = self.target_detectors[label].get_most_probable_target(timestamp = timestamp)
            print(f'get_most_probable_target time for hand {label} : {(time.time()-t)*1000:.2f} ms')
        t = time.time()
        objs = self.objects.keys()
        for olabel in objs:
            target_info = (False, None, None)
            for dlabel in target_detector_labels:
                if targets[dlabel] is not None:
                    if olabel == targets[dlabel].label:
                        target_info=(True, self.hands[dlabel], targets[dlabel][olabel])
                    self.objects[olabel].set_target_info(target_info)
        print(f'set_target_info time : {(time.time()-t)*1000:.2f} ms')
        # print(f'fetch_all_targets time : {(time.time()-t)*1000:.2f} ms')

    def evaluate_grasping_intention(self):
        hands = self.hands.copy().values
        objs = self.objects.values()
        for obj in objs():
            for hand in hands:
                if hand.label=='right':
                    obj.is_targeted_by(hand)


    def update_hands_time(self, timestamp = None):
        if timestamp is None:
            now = time.time()
        else:
            now = timestamp
        self.elapsed_hands= now - self.time_hands 
        hands = self.hands.copy().values()
        for hand in hands:
            hand.set_timestamp(now)
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
            # print(f"get_{hand.label}_data: {data[hand.label + '_hand']}")
        return data

    def get_hands_rendering_data(self):
        data = {}
        for hand in self.hands.values():
            data[hand.label + '_hand'] = hand.get_rendering_data()
        return data
    
    def get_objects_data(self):
        data = {}
        for obj in self.objects.values():
            data[obj.label] = obj.get_trajectory()
            # print(f"get_{obj.label}_data: {data[obj.label ]}")
        return data
    
    def get_objects_rendering_data(self):
        data = {}
        for obj in self.objects.values():
            data[obj.label] = obj.get_rendering_data()
        return data
    
    def get_target_data(self):
        data = {}
        for hand in self.hands.values():
            data[hand.label + '_hand'] = hand.get_target_data()
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
        
    def __init__(self, cam_data, name='Grasping experiment',   video_rendering_options = _DEFAULT_VIDEO_RENDERING_OPTIONS, scene_rendering_options = _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS, dataset='ycbv', plotter=None, fps = 40) -> None:
        super().__init__(cam_data, name, video_rendering_options, scene_rendering_options, dataset=dataset, plotter=plotter, fps=fps)
    
    def render(self, img):
        # self.compute_distances()
        self.update_scene_time()
        t = time.time()
        hands = self.hands.copy().values()
        objs = self.objects.copy().values()
        print(f'get hands and objects time : {(time.time()-t)*1000:.2f} ms')
        t = time.time()
        for hand in hands:
            hand.render(img)
        print(f'hand render time : {(time.time()-t)*1000:.2f} ms')
        t = time.time()
        for obj in objs:
            obj.render(img)
        print(f'obj render time : {(time.time()-t)*1000:.2f} ms')
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
    
    def __init__(self, cam_data, name='Grasping experiment',   video_rendering_options = _DEFAULT_VIDEO_RENDERING_OPTIONS, scene_rendering_options = _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS, dataset = None) -> None:
        super().__init__(cam_data, name, video_rendering_options, scene_rendering_options, detect_grasping=False, dataset=dataset)
    
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


class AnalysisScene(Scene):
    _DEFAULT_VIDEO_RENDERING_OPTIONS=dict(write_fps = True, 
                                    draw_hands=True, 
                                    draw_objects=False, 
                                    write_hands_pos = False, 
                                    write_objects_pos = False,
                                    render_objects=True)
    
    _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS = dict(is_displayed = True,
                                                    draw_grid = True,
                                                    show_velocity_cone = True)
    
    def __init__(self, cam_data, name='Grasping experiment',   video_rendering_options = _DEFAULT_VIDEO_RENDERING_OPTIONS, scene_rendering_options = _DEFAULT_VIRTUAL_SCENE_RENDERING_OPTIONS, fps=30) -> None:
        super().__init__(cam_data, name, video_rendering_options, scene_rendering_options, detect_grasping=True, fps=fps)

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
        
