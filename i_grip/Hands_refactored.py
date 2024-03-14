import numpy as np
import pandas as pd
import trimesh as tm
import cv2
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from i_grip.utils2 import *
from i_grip import Hands3DDetectors as hd
from typing import  Mapping
# from i_grip.HandDetectors2 import HandPrediction


# from sklearn.linear_model import Ridge
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
    
class GraspingHandTrajectory(Trajectory):
    
    DEFAULT_DATA_KEYS = [ 'Timestamps', 'x', 'y', 'z', 'Extrapolated']
    DEFAULT_ATTRIBUTES  = dict(timestamp=True, filtered_position=True)
    
    def __init__(self, state = None, headers_list = DEFAULT_DATA_KEYS, attributes_dict=DEFAULT_ATTRIBUTES, file = None, dataframe=None, fit_method = 'np_poly', limit_size=None) -> None:
        super().__init__(state, headers_list, attributes_dict, file, dataframe, limit_size)
        self.poly_coeffs = {}
        self.polynomial_function = None
        self.was_fitted = False
        if fit_method == 'np_poly':
            self.fit = self.polynomial_fit
            self.extrapolate = self.np_poly_extrapolate
            
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, headers_list=DEFAULT_DATA_KEYS, attributes_dict=None, limit_size=None):
        return cls(dataframe = df, headers_list=headers_list, attributes_dict=attributes_dict, limit_size=limit_size)
    
    def __next__(self):
        if self.current_line_index < len(self.data):
            row = self.data.iloc[self.current_line_index]
            # print(f'data : {self.data}')
            # print('row', row)
            self.current_line_index+=1
            return Position(np.array([row['x'], row['y'], row['z']]), display='cm', swap_y=True), row['Timestamps']
        else:
            raise StopIteration
        
    def __getitem__(self, index):
        if index < len(self.data):
            row = self.data.iloc[index]
            return Position(np.array([row['x'], row['y'], row['z']]), display='cm', swap_y=True), row['Timestamps']
        else:
            raise IndexError('Index out of range')
    
    def polynomial_fit(self, nb_points, degree=2):
        # find polynomial fit for the last nb_points of the trajectory
        if len(self.data) < nb_points:
            print('Not enough data points to find a polynomial fit')
        else:
            last_points = self.data.iloc[-nb_points:]
            t = last_points['Timestamps'].values
            x = last_points['x'].values
            y = last_points['y'].values
            z = last_points['z'].values
            # self.poly_coeffs = np.polynomial.polynomial.Polynomial.fit(t, xyz_data, degree)
            self.poly_coeffs['x'] = np.polynomial.polynomial.Polynomial.fit(t, x, degree)
            self.poly_coeffs['y'] = np.polynomial.polynomial.Polynomial.fit(t, y, degree)
            self.poly_coeffs['z'] = np.polynomial.polynomial.Polynomial.fit(t, z, degree)
            self.was_fitted = True
            # self.poly_coeffs['x'] = np.polynomial.chebyshev.chebfit(t, x, degree)
            # self.poly_coeffs['y'] = np.polynomial.chebyshev.chebfit(t, y, degree)
            # self.poly_coeffs['z'] = np.polynomial.chebyshev.chebfit(t, z, degree)
            
            # self.polynomial_function = np.polynomial.polynomial.polyval(t, self.poly_coeffs)
    
    
    # def np_poly_extrapolate(self, timestamps):
    #     # extrapolate the trajectory using a polynomial fit
    #     if not self.was_fitted:
    #         # print('No polynomial fit found, please use polynomial_fit() first')
    #         x = self.data.loc[len(self.data)-1]['x']
    #         y = self.data.loc[len(self.data)-1]['y']
    #         z = self.data.loc[len(self.data)-1]['z']
    #         # x = np.polynomial.chebyshev.chebval(timestamps, self.poly_coeffs['x'])
    #         # y = np.polynomial.chebyshev.chebval(timestamps, self.poly_coeffs['y'])
    #         # z = np.polynomial.chebyshev.chebval(timestamps, self.poly_coeffs['z'])
    #         return np.array([x,y,z])
    #     else:
    #         x = self.poly_coeffs['x'](timestamps)
    #         y = self.poly_coeffs['y'](timestamps)
    #         z = self.poly_coeffs['z'](timestamps)
    #     return np.array([x,y,z])
    def np_poly_extrapolate(self, timestamps):
        # extrapolate the trajectory using a polynomial fit
        if not self.was_fitted:
            # print('No polynomial fit found, please use polynomial_fit() first')
            x = self.data.loc[len(self.data)-1]['x']
            y = self.data.loc[len(self.data)-1]['y']
            z = self.data.loc[len(self.data)-1]['z']
            # x = np.polynomial.chebyshev.chebval(timestamps, self.poly_coeffs['x'])
            # y = np.polynomial.chebyshev.chebval(timestamps, self.poly_coeffs['y'])
            # z = np.polynomial.chebyshev.chebval(timestamps, self.poly_coeffs['z'])
            return np.array([x,y,z])
        else:
            print(f'timestamps : {timestamps}')
            x = self.poly_coeffs['x'](np.array(timestamps))
            y = self.poly_coeffs['y'](np.array(timestamps))
            z = self.poly_coeffs['z'](np.array(timestamps))
            res = np.array([x,y,z]).T
            print(f'extrapolated res : {res}')  
        return res

    
    def compute_last_derivatives(self, nb_points=None):
        # compute derivatives of the last nb_points of the trajectory
        if nb_points is None:
            print('Using all data points to compute derivatives')
            last_points = self.data
        elif len(self.data) < nb_points:
            print(f'Not enough data points to compute derivatives, using all data points ({len(self.data)})')
            last_points = self.data
        else:
            last_points = self.data.iloc[-nb_points:]
        x = last_points['x'].values
        y = last_points['y'].values
        z = last_points['z'].values
        t = last_points['Timestamps'].values
        # print(t)
        # print(x)
        # print(type(t))
        # print(type(x))  
        # print(f'last_points : {last_points}')
        max_acc = 2
        t = np.array(t)
        if len(t) < 2:
            velocity = np.array([0,0,0])
            acceleration = np.array([0,0,0])
        elif len(t) < 3:
            velocity = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])/(t[1]-t[0])
            acceleration = np.array([0,0,0])
        else:
            positions = np.array([x,y,z]).T
            if len(t) <= 5:
                acc = 2
            elif len(t) <= 8:
                acc = 4
            elif len(t) <= 12:
                acc = 6
            else:
                acc = 8
            acc = min(acc, max_acc)
            velocity = findiff_diff(t, positions, diff_order=1, diff_acc=acc)
            acceleration = np.array([0,0,0])
        return velocity, acceleration
    
    def get_xyz_data(self):
        xyz_observed_list = []
        xyz_extrapolated_list = []
        
        last_points = self.data.iloc[-100:]
        # print(f'last_points : {last_points}')
        for index, row in last_points.iterrows():
            xyz = np.array([-row['x'], row['y'], row['z']])
            if row['Extrapolated']:
                xyz_extrapolated_list.append(xyz)
            else:                
                xyz_observed_list.append(xyz)
        return xyz_observed_list, xyz_extrapolated_list
    
    def __repr__(self) -> str:
        return super().__repr__()


class GraspingHand(Entity):
    MAIN_DATA_KEYS=GraspingHandTrajectory.DEFAULT_DATA_KEYS
    
    def __init__(self, input, label = None, timestamp=None, plotter = None)-> None:
        super().__init__(timestamp=timestamp)
        self.label = label
        self.plotter = plotter
        print(f'new hand : {self.label}')
        print(type(input))
        if input is None:
            raise ValueError('input must be provided')
        else:
            if isinstance(input, hd.HandPrediction):
                print('input is a HandPrediction')
                self.detected_hand = input
                self.state = GraspingHandState.from_hand_detection(input, timestamp = timestamp)
                if self.label is None:
                    self.label = input.label         
                # self.trajectory = GraspingHandTrajectory.from_state(self.state)
            elif isinstance(input, np.ndarray):
                self.state = GraspingHandState.from_position(input, timestamp = timestamp)            
                # self.trajectory = GraspingHandTrajectory.from_state(self.state)
            
            elif isinstance(input, pd.DataFrame):
                # self.trajectory = GraspingHandTrajectory.from_dataframe(input)
                # first_position, first_timestamp = self.trajectory[0]
                # print(f'next hand position : {first_position, first_timestamp}')
                # self.state = GraspingHandState.from_position(first_position, timestamp = first_timestamp)
                print(f'buiding hand from dataframe : {input}')
                self.state = GraspingHandState.from_dataframe(input)
                
            else:   
                self.state = None
        
        self.new= True
        self.visible = True
        self.invisible_time = 0
        self.max_invisible_time = 0.3
        
 
        # self.update_trajectory(timestamp)
        self.show_label = True
        self.show_xyz = True
        self.show_roi = True
        self.margin = 10  # pixels
        self.font_size = 1
        self.font_thickness = 1
        self.label_text_color = (88, 205, 54) # vibrant green
        self.font_size_xyz = 0.5
    
        ### define mesh
        self.impact_locations = {}
        self.hand_pos_obj_frame = {}
        self.impact_locations_list = {}
        self.rays_vizualize_list = {}
        self.full_hand = False
        self.define_mesh_representation()
        self.update_mesh()
        self.most_probable_target = None
        self.targets_data = pd.DataFrame(columns = ['Timestamp', 'label', 'grip', 'time_before_impact','ratio'])

    @classmethod
    def from_trajectory(cls, trajectory):
        return cls(trajectory = trajectory)
    
    # def build_from_hand_prediction(self, hand_prediction):
    #     self.__dict__= hand_prediction.__dict__
    #     if isinstance(hand_prediction, i_grip.HandDetectors.HandPrediction):
    #         self.draw = hand_prediction.draw
    #     self.base_dict = hand_prediction.__dict__
        
    def build_from_scratch(self, label):        
        self.label = label
        self.state = GraspingHandState()
        
    def define_mesh_representation(self):
        self.mesh_origin = tm.primitives.Sphere(radius = 20)
        if self.label == 'right':
            self.mesh_origin.visual.face_colors = self.color = [255,0,0,255]
            self.plot_color = 'red'
            self.extrapolated_trajectory_color = [0,100,0,100]
            self.future_color = [0,255,0,255]
            self.text_color = [0,0,255]
        else:  
            self.mesh_origin.visual.face_colors = self.color = [0,0,100,255]
            self.extrapolated_trajectory_color = [50,50,50,50]
            self.text_color=[100,0,0]
            self.plot_color = 'blue'
            self.future_color = [0,255,255,255]
            
        if self.full_hand:
            self.mesh_key_points = [tm.primitives.Sphere(radius = 7) for i in range(21)]
            mp_specs = solutions.drawing_styles.get_default_hand_landmarks_style()
            for i in range(21):                
                drawing_spec = mp_specs[i] if isinstance(
                    mp_specs, Mapping) else mp_specs
                color = drawing_spec.color
                self.mesh_key_points[i].visual.face_colors = [color[0], color[1], color[2], 255]
            self.key_points_key_points_connections=solutions.hands.HAND_CONNECTIONS
            self.key_points_connections_path = tm.load_path(np.zeros((2,2,3)))
    
    def get_keypoints_representation(self):
        return self.key_points_mesh_transforms, self.key_points_connections_path
    
    def update(self, detected_hand):
        self.detected_hand = detected_hand
        self.state.update(detected_hand)
        
    def update_from_trajectory(self, index = None):
        #T0D0 : modifier state pour tout faire les majs en interne
        if index is None:         
            pos, timestamp= self.state.__next__()
        else:
            pos, timestamp = self.state[index]
        self.state.update(pos)
        self.state.propagate_all(timestamp)
        # self.state.propagate(timestamp)
        self.set_mesh_updated(False)
        
    def update_from_trajectory_saved(self, index = None):
        if index is None:         
            pos, timestamp= self.trajectory.__next__()
        else:
            pos, timestamp = self.trajectory[index]
        self.state.update(pos)
        self.state.propagate(timestamp)
        self.set_mesh_updated(False)
        
    def propagate(self, timestamp=None):
        # print(f'timestamp : {timestamp}')
        t = time.time()
        self.state.propagate_all(timestamp)
        print(f'propagate all hand time : {(time.time()-t)*1000:2f} ms')
        # self.state.propagate(timestamp)
        self.set_mesh_updated(False)
        # self.update_trajectory()
        # self.update_meshes()
              
    def update_mesh(self):
        if  self.was_mesh_updated():
            return
        self.mesh_position = Position(self.state.position_filtered*np.array([-1,1,1]))
        # self.mesh_position = Position(self.state.velocity_filtered*np.array([-1,1,1])+np.array([0,0,500]))
        self.mesh_transform= tm.transformations.translation_matrix(self.mesh_position.v)
        if self.full_hand:
            # TODO : try a more efficient np formulation
            t = time.time()
            self.key_points_mesh_transforms = [tm.transformations.translation_matrix(self.state.world_landmarks[i]*np.array([-1,1,1])) for i in range(21)]
            print(f'key_points_mesh_transforms time : {(time.time()-t)*1000:2f} ms')
            # print(f'connections : {self.key_points_key_points_connections}')
            # start_indexes = [connection[0] for connection in self.key_points_key_points_connections]
            # end_indexes = [connection[1] for connection in self.key_points_key_points_connections]
            # for connection in self.key_points_key_points_connections:
            #     start_indexes
            self.key_points_connections_path_starts = np.vstack([self.state.world_landmarks[connection[0]]*np.array([-1,1,1]) for connection in self.key_points_key_points_connections])
            self.key_points_connections_path_ends = np.vstack([self.state.world_landmarks[connection[1]]*np.array([-1,1,1]) for connection in self.key_points_key_points_connections])
            self.key_points_connections_path = tm.load_path(np.hstack((self.key_points_connections_path_starts, self.key_points_connections_path_ends)).reshape(-1, 2, 3)) 
        self.set_mesh_updated(True)
    
    def get_keypoints_representation(self):
        return self.key_points_mesh_transforms, self.key_points_connections_path
    
    def get_mesh_position(self):
        return self.mesh_position.v
    
    def get_trajectory_points(self):
        return self.state.trajectory.get_xyz_data()
    
    def get_movement_direction(self):
        return self.state.get_movement_direction()
    
    def get_scalar_velocity(self):
        return self.state.scalar_velocity
    
    def get_future_trajectory_points(self):
        self.future_points = self.state.get_future_trajectory_points()
        # if not hasattr(self, 'dotruc'):
        #     self.dotruc = 0
        # if self.dotruc <20:            
        #     self.future_points = self.state.get_future_trajectory_points(timespand)
        #     self.dotruc +=1
        # print(f'doa truc : {self.dotruc}')
        # self.future_points = [self.mesh_position.v + np.array([10*i,2*i**2,0*i]) for i in range(25)]
        return self.future_points
    
    
    def fetch_targets(self):
        self.most_probable_target, targets = self.target_detector.get_most_probable_target(self.elapsed)
        # print(f'{self.label} hand most probable target : {self.most_probable_target}')
        # if self.most_probable_target is not None:
        #     self.targets_data.loc[len(self.targets_data)] = [timestamp]+list(self.most_probable_target.get_info())
        # else:
        #     self.targets_data.loc[len(self.targets_data)] = [timestamp, '-', '-', '-', '-']
        return targets
    
    def render(self, img):
        # # Draw the hand landmarks.
        # # displayed_landmarks = self.state.landmarks
        # t = time.time()
        # displayed_landmarks = self.detected_hand.get_landmarks()
        # landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        # # landmarks_proto.landmark.extend([
        # #                                     landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in displayed_landmarks])
        # landmarks_proto.landmark.extend([
        #                                     landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in displayed_landmarks])
        # solutions.drawing_utils.draw_landmarks(
        #                                     img,
        #                                     landmarks_proto,
        #                                     solutions.hands.HAND_CONNECTIONS,
        #                                     solutions.drawing_styles.get_default_hand_landmarks_style(),
        #                                     solutions.drawing_styles.get_default_hand_connections_style())
        # print(f'draw_landmarks tie : {(time.time()-t)*1000:2f} ms')
        # if self.show_label:
        #     # Get the top left corner of the detected hand's bounding box.
        #     text_x = int(min(displayed_landmarks[:,0]))
        #     text_y = int(min(displayed_landmarks[:,1])) - self.margin

        #     # Draw handedness (left or right hand) on the image.
        #     # cv2.putText(img, f"{self.handedness[0].category_name}",
        #     #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #     #             self.font_size, self.handedness_text_color, self.font_thickness, cv2.LINE_AA)
            
        #     cv2.putText(img, f"{self.label}",
        #                 (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #                 self.font_size, self.label_text_color, self.font_thickness, cv2.LINE_AA)
        # if self.show_roi:
        #     cv2.rectangle(img, (self.detected_hand.roi[0],self.detected_hand.roi[1]),(self.detected_hand.roi[2],self.detected_hand.roi[3]),self.label_text_color)

        # if self.show_xyz:
        #     # Get the top left corner of the detected hand's bounding box.z
            
        #     #print(f"{self.label} --- X: {self.xyz[0]/10:3.0f}cm, Y: {self.xyz[0]/10:3.0f} cm, Z: {self.xyz[0]/10:3.0f} cm")
        #     if len(img.shape)<3:
        #         height, width = img.shape
        #     else:
        #         height, width, _ = img.shape
        #     x_coordinates = displayed_landmarks[:,0]
        #     y_coordinates = displayed_landmarks[:,1]
        #     x0 = int(max(x_coordinates) * width)
        #     y0 = int(max(y_coordinates) * height) + self.margin

        #     # Draw handedness (left or right hand) on the image.
        #     cv2.putText(img, f"X:{self.state.position_filtered.x/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_DUPLEX, self.font_size_xyz, (20,180,0), self.font_thickness, cv2.LINE_AA)
        #     cv2.putText(img, f"Y:{self.state.position_filtered.y/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_DUPLEX, self.font_size_xyz, (255,0,0), self.font_thickness, cv2.LINE_AA)
        #     cv2.putText(img, f"Z:{self.state.position_filtered.z/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_DUPLEX, self.font_size_xyz, (0,0,255), self.font_thickness, cv2.LINE_AA)

        GraspingHand.render_hand(img, self.label, self.detected_hand.get_landmarks(), self.detected_hand.roi, self.state.position_filtered, self.show_label, self.show_xyz, self.show_roi, self.margin, self.font_size, self.font_thickness, self.font_size_xyz, self.label_text_color)
    
    def render_hand(img, label = None, displayed_landmarks = None, roi=None, position=None, show_label = True, show_xyz = True, show_roi = True, margin = 10, font_size = 1, font_thickness = 1, font_size_xyz = 0.5, label_text_color = (88, 205, 54)):
        if displayed_landmarks is not None:
            # Draw the hand landmarks.
            landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            landmarks_proto.landmark.extend([
                                                landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in displayed_landmarks])
            solutions.drawing_utils.draw_landmarks(
                                                img,
                                                landmarks_proto,
                                                solutions.hands.HAND_CONNECTIONS,
                                                solutions.drawing_styles.get_default_hand_landmarks_style(),
                                                solutions.drawing_styles.get_default_hand_connections_style())
        if show_label:
            # Get the top left corner of the detected hand's bounding box.
            text_x = int(min(displayed_landmarks[:,0]))
            text_y = int(min(displayed_landmarks[:,1])) - margin

            # Draw handedness (left or right hand) on the image.
            cv2.putText(img, f"{label}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        font_size, label_text_color, font_thickness, cv2.LINE_AA)
        if show_roi and roi is not None:
            cv2.rectangle(img, (roi[0],roi[1]),(roi[2],roi[3]),label_text_color)

        if show_xyz and position is not None:
            # Get th e top left corner of the detected hand's bounding box.z
            
            #print(f"{label} --- X: {xyz[0]/10:3.0f}cm, Y: {xyz[0]/10:3.0f} cm, Z: {xyz[0]/10:3.0f} cm")
            if len(img.shape)<3:
                height, width = img.shape
            else:
                height, width, _ = img.shape
            x_coordinates = displayed_landmarks[:,0]
            y_coordinates = displayed_landmarks[:,1]
            x0 = int(max(x_coordinates) * width)
            y0 = int(max(y_coordinates) * height) + margin

            # Draw handedness (left or right hand) on the image.
            cv2.putText(img, f"X:{position.x/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_DUPLEX, font_size_xyz, (20,180,0), font_thickness, cv2.LINE_AA)
            cv2.putText(img, f"Y:{position.y/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_DUPLEX, font_size_xyz, (255,0,0), font_thickness, cv2.LINE_AA)
            cv2.putText(img, f"Z:{position.z/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_DUPLEX, font_size_xyz, (0,0,255), font_thickness, cv2.LINE_AA)
    
    def get_rendering_data(self):
        data={}
        data['displayed_landmarks'] = self.detected_hand.get_landmarks()
        data['roi'] = self.detected_hand.roi
        data['position'] = self.state.position_filtered
        data['font_size'] = self.font_size
        data['font_thickness'] = self.font_thickness
        data['font_size_xyz'] = self.font_size_xyz
        data['label_text_color'] = self.label_text_color
        data['label'] = self.label
        data['show_label'] = self.show_label
        data['show_xyz'] = self.show_xyz
        data['show_roi'] = self.show_roi
        data['margin'] = self.margin
        return data

    def get_target_data(self):
        return self.targets_data
        
class GraspingHandState(State):
    def __init__(self,  position=None, normalized_landmarks=None, world_landmarks = None,  timestamp = None, trajectory = None) -> None:
        super().__init__()
        self.position_raw = Position(position)
        self.normalized_landmarks = normalized_landmarks
        print(f'buiding hand state with world_landmarks : {world_landmarks}')
        self.world_landmarks = world_landmarks
        # if landmarks is None:
        #     self.landmarks = np.zeros((21,3))
        # else:
        #     self.landmarks = landmarks
        self.new_position = self.position_raw
        self.new_normalized_landmarks = self.normalized_landmarks
        self.new_world_landmarks = self.world_landmarks
        
        self.velocity_raw = np.array([0,0,0])
        self.scalar_velocity = 0
        self.normed_velocity = np.array([0,0,0])
        self.position_filtered = self.position_raw
        self.velocity_filtered = self.velocity_raw
        self.filter_position, self.filter_velocity = Filter.both('position')
        
        if normalized_landmarks is not None:
            self.normalized_landmarks_velocity = np.zeros((21,3))
            self.normalized_landmarks_filtered = self.normalized_landmarks
            self.normalized_landmarks_velocity_filtered = self.normalized_landmarks_velocity            
            self.filter_normalized_landmarks, self.filter_normalized_landmarks_velocity= Filter.both('normalized_landmarks')
            
        self.future_points=[self.position_raw.v]
            
        if timestamp is None:
            self.last_timestamp = time.time()
        else:
            self.last_timestamp = timestamp
        self.scalar_velocity_threshold = 20 #mm/s
        if trajectory is None:
            # self.trajectory = GraspingHandTrajectory.from_state(self)
            self.trajectory = GraspingHandTrajectory.from_state(self, limit_size=20)
        else:
            self.trajectory = trajectory
        self.extraplation_count = 0
        self.set_updated(True)
        self.set_propagated(False)
        
    @classmethod
    def from_hand_detection(cls, hand_detection: hd.HandPrediction, timestamp = 0):
        return cls(hand_detection.position, hand_detection.normalized_landmarks, hand_detection.world_landmarks, timestamp)
    
    @classmethod
    def from_position(cls, position: Position, timestamp = 0):
        return cls(position, timestamp=timestamp)

    # @classmethod   
    # def from_trajectory(cls, trajectory: Trajectory, timestamp = 0):
    #     return cls(trajectory.current_state.position, trajectory.current_state.normalized_landmarks, timestamp, trajectory)

    @classmethod
    def from_dataframe(cls, df:pd.DataFrame):
        # first_row = df.iloc[0]
        # first_position = Position(np.array([first_row['x'], first_row['y'], first_row['z']]), display='cm', swap_y=True)
        # first_timestamp = first_row['Timestamps']
        print(f'buiding hand state from dataframe : {df}')
        trajectory = GraspingHandTrajectory.from_dataframe(df)
        first_position, first_timestamp = trajectory[0]
        print(f'next hand position : {first_position, first_timestamp}')
        
        return cls(first_position, timestamp=first_timestamp, trajectory=trajectory)
    
    def update_position(self, position):
        self.new_position = Position(position)
        
    def update_normalized_landmarks(self, normalized_landmarks):
        self.new_normalized_landmarks = normalized_landmarks
    
    def update_world_landmarks(self, world_landmarks):
        self.new_world_landmarks = world_landmarks
    
    def update(self, new_input):
        if isinstance(new_input, hd.HandPrediction):
            self.update_position(new_input.position)
            self.update_normalized_landmarks(new_input.normalized_landmarks)
            self.update_world_landmarks(new_input.world_landmarks)
        elif isinstance(new_input, Position):
            self.update_position(new_input)
        elif new_input is None:
            self.update_normalized_landmarks(new_input)
        else:
            print(f'weird input : {new_input}')
        self.set_updated(True)
        self.set_propagated(False)
        
    def propagate(self, timestamp):
        elapsed = timestamp - self.last_timestamp        
        self.last_timestamp = timestamp
        self.propagate_position(elapsed)
        if self.normalized_landmarks is not None:
            self.propagate_normalized_landmarks(elapsed)
        if self.world_landmarks is not None:
            self.propagate_world_landmarks(elapsed) 
        self.set_updated(False)
        self.set_propagated(True)
        print(f'propagated hand position : {self.position_raw, self.last_timestamp}')
        print(f'propgated hand world landmarks : {self.world_landmarks}')
        
    def propagate_position(self, elapsed):
        if not  self.was_updated():
            next_position = self.position_raw
        else:
            next_position = self.new_position
            next_position_filtered = Position(self.filter_position.apply(next_position.v))
            if elapsed >0:
                self.velocity_raw = (next_position_filtered.v - self.position_filtered.v)/elapsed
            self.position_filtered = next_position_filtered
            
            self.velocity_filtered = self.filter_velocity.apply(self.velocity_raw)
            
            self.scalar_velocity = np.linalg.norm(self.velocity_filtered)
            if self.scalar_velocity != 0:
                if self.scalar_velocity > self.scalar_velocity_threshold:
                    self.normed_velocity = self.velocity_filtered/self.scalar_velocity
                else:
                    self.normed_velocity = self.normed_velocity*98/100 + self.velocity_filtered/self.scalar_velocity*2/100
            else:
                self.normed_velocity = np.array([0,0,0])
                
            self.position_raw = next_position
    
    def propagate_all(self, timestamp):
        
        elapsed = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        self.propagate_only_position()
        self.apply_filter_position()
        self.compute_velocity()
        if self.normalized_landmarks is not None:
            self.propagate_normalized_landmarks(elapsed)
        if self.world_landmarks is not None:
            self.propagate_world_landmarks(elapsed) 
        self.compute_future_points()
        self.set_updated(False)
        self.set_propagated(True)
    
    def propagate_only_position(self):
        if self.was_updated():                   
            self.position_raw = self.new_position
            self.trajectory.add(self)
            self.trajectory.fit(10)
            self.extraplation_count = 0
            # print('was_updated')
        elif self.extrapolation_allowed():
            # print('was extrapolated')
            extrapolated_position = self.trajectory.extrapolate(self.last_timestamp)
            # print(f'extrapolated_position : {extrapolated_position}')
            self.position_raw = Position(extrapolated_position)
            self.trajectory.add(self, extrapolated=True)
            self.extraplation_count += 1
            
    def extrapolation_allowed(self):
        return self.extraplation_count < 5
            
            
    def compute_future_points(self, nb_steps=10, timestep=0.1):
        vel_unit = 50
        nb_steps = int(self.scalar_velocity/vel_unit)
        # nb_steps = 0
        timestamps = [self.last_timestamp + timestep*i for i in range(nb_steps)]
        self.future_points=[]
        if nb_steps == 0:
            self.future_points.append(self.position_filtered.v*np.array([-1,1,1]))
        future_points = self.trajectory.extrapolate(timestamps)
        self.future_points += [future_point*np.array([-1,1,1]) for future_point in future_points]
        # for timestamp in timestamps:
        #     future_point = self.trajectory.extrapolate(timestamp)
        #     # print(f'future_point : {future_point}')
        #     self.future_points.append(future_point*np.array([-1,1,1]))
            
    def get_future_trajectory_points(self):
        return self.future_points
    
    def apply_filter_position(self):
        self.position_filtered = Position(self.filter_position.apply(self.position_raw.v))
            
    def compute_velocity(self):
        if not self.was_updated():
            return
        # print(f'position_filtered : {self.position_filtered.v}')
        # print(f'trajectory : {self.trajectory}')
        self.velocity_raw, self.acceleration_raw = self.trajectory.compute_last_derivatives(2)
        self.velocity_filtered = self.filter_velocity.apply(self.velocity_raw)
        self.scalar_velocity = np.linalg.norm(self.velocity_filtered)
        if self.scalar_velocity != 0:
            if self.scalar_velocity > self.scalar_velocity_threshold:
                self.normed_velocity = self.velocity_filtered/self.scalar_velocity
            else:
                self.normed_velocity = self.normed_velocity*98/100 + self.velocity_filtered/self.scalar_velocity*2/100
        else:
            self.normed_velocity = np.array([0,0,0])
            
    
        
    def propagate_normalized_landmarks(self, elapsed):
        if not  self.was_updated():
            # next_normalized_landmarks = self.normalized_landmarks + elapsed*self.normalized_landmarks_velocity
            next_normalized_landmarks = self.normalized_landmarks
            self.normalized_landmarks_velocity = self.normalized_landmarks_velocity
        else:
            next_normalized_landmarks = self.new_normalized_landmarks
            if elapsed >0:
                self.normalized_landmarks_velocity = (self.new_normalized_landmarks - self.normalized_landmarks)/elapsed
            
        self.normalized_landmarks_filtered = self.filter_normalized_landmarks.apply(next_normalized_landmarks)
        self.normalized_landmarks_velocity_filtered = self.filter_normalized_landmarks_velocity.apply(self.normalized_landmarks_velocity)
            
        self.normalized_landmarks = next_normalized_landmarks
    
    def propagate_world_landmarks(self, elapsed):
        self.world_landmarks = self.new_world_landmarks
    
    def get_movement_direction(self):
        point_down_factor = -0.3
        vdir = self.normed_velocity + np.array([0,point_down_factor,0])
        if np.linalg.norm(vdir) != 0:
            vdir = vdir / np.linalg.norm(vdir)
        return vdir

        
    
    def as_list(self, timestamp=True, position=False, normalized_landmarks=False, velocity=False, normalized_landmarks_velocity=False, filtered_position=False, filtered_velocity=False, filtered_normalized_landmarks=False, filtered_normalized_landmarks_velocity=False, normalized_velocity=False, scalar_velocity=False):
        repr_list = []
        if timestamp:
            repr_list.append(self.last_timestamp)
        if position:
            repr_list += self.position_raw.as_list()
        if normalized_landmarks:
            repr_list += self.normalized_landmarks.flatten().tolist()
        if velocity:
            repr_list += self.velocity_raw.tolist()
        if normalized_landmarks_velocity:
            repr_list += self.normalized_landmarks_velocity.flatten().tolist()
        if filtered_position:
            repr_list += self.position_filtered.as_list()
        if filtered_velocity:
            repr_list += self.velocity_filtered.tolist()
        if filtered_normalized_landmarks:
            repr_list += self.normalized_landmarks_filtered.flatten().tolist()
        if filtered_normalized_landmarks_velocity:
            repr_list += self.normalized_landmarks_velocity_filtered.flatten().tolist()
        if normalized_velocity:
            repr_list += self.normed_velocity.tolist()
        if scalar_velocity:
            repr_list.append(self.scalar_velocity)
        return repr_list

    def __next__(self):
        return self.trajectory.__next__()

    def __getitem__(self, index):
        return self.trajectory[index]
    
    def __str__(self) -> str:
        return f'HandState : {self.position_filtered}'
    
    def __repr__(self) -> str:
        return self.__str__()