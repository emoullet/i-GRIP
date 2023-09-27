#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import cv2
import subprocess
import pandas as pd
import os
from i_grip.HandDetectors2 import HandPrediction
from i_grip.Filters import LandmarksSmoothingFilter

class Position:
    def __init__(self, input = None, display='mm', swap_y = False) -> None:
        '''Position in millimeters'''
        # vect[1] = -vect[1]
        if input is None:
            vect = np.array([0,0,0])
        elif isinstance(input, Position):
            vect = input.v
        elif isinstance(input, np.ndarray):
            #check if input is a 3D vector
            if input.shape == (3,):
                vect = input
            else:
                raise ValueError('Position must be initialized with a 3D vector')
        elif isinstance(input, list):
            if len(input)==3:
                vect = np.array(input)
            else:
                raise ValueError('Position must be initialized with a list of 3 elements')
            vect = np.array(input)
        else:
            raise TypeError('Position must be initialized with a list, a numpy array or a Position object')
        
        self.x = vect[0]
        self.y = vect[1]
        self.z = vect[2]
        self.v = vect
        self.display = display  
        self.ve = np.hstack((self.v,1)) 

    def __str__(self):
        if self.display == 'cm':
            return str(self.v/10)+' (in cm)'
        else:
            return str(self.v)+' (in mm)'

    def __mul__(self, other):
        if isinstance(other, Position):
            return Position(self.v*other.v)
        elif isinstance(other, np.ndarray):
            if other.shape == (3,):
                return Position(self.v*other)
            else:
                raise ValueError('Position can only be multiplied by a 3D vector')
        elif isinstance(other, (int, float)):
            return Position(self.v*other)
        elif isinstance(other, R):
            return Position(other.apply(self.v))
        elif isinstance(other, (list, tuple)):
            if len(other)==3:
                return Position(self.v*np.array(other))
            else:
                raise ValueError('Position can only be multiplied by a list of 3 elements')
        else:
            raise TypeError('Position can only be multiplied by a Position, a numpy array, a list or a float')
    
    def __add__(self, other):
        if isinstance(other, Position):
            return Position(self.v+other.v)
        else :
            try:
                return self + Position(other)
            except:
                raise TypeError('Position can only be added to a Position, a numpy array or a list')
        
    
    def __call__(self):
        return self.v
    
    def as_list(self):
        return self.v.tolist()
    @staticmethod
    def distance(p1 : 'Position', p2 : 'Position'):
        return np.linalg.norm(p1.v - p2.v)
    
    def as_list(self):
        return self.v.tolist()

class Orientation:
    def __init__(self, mat) -> None:
        self.r = R.from_matrix(mat)
        self.v = self.r.as_euler('xyz')
        self.x = self.v[0]
        self.y = self.v[1]
        self.z = self.v[2]
        self.q = self.r.as_quat()
        self.qx = self.q[0]
        self.qy = self.q[1]
        self.qz = self.q[2]
        self.qw = self.q[3]

    @classmethod
    def from_quat(cls, quat):
        r = R.from_quat(quat)
        return cls(r.as_matrix())
    
    def __str__(self):
        return str(self.v)
    
    def as_list(self):
        return self.q.tolist()
    
class Pose:
    def __init__(self, tensor_or_mat, position_factor=1, orientation_factor=1) -> None:
        self.min_cutoffs = {'position' : 0.001, 'orientation' : 0.0001}
        self.betas = {'position' : 100, 'orientation' : 10}
        self.derivate_cutoff = {'position' : 0.1, 'orientation' : 1}
        attributes = ('position', 'orientation')
        self.filters={}
        self.position_factor = position_factor
        self.orientation_factor = orientation_factor
        for key in attributes:
            self.filters[key] = LandmarksSmoothingFilter(min_cutoff=self.min_cutoffs[key], beta=self.betas[key], derivate_cutoff=self.derivate_cutoff[key], disable_value_scaling=True)
        self.update(tensor_or_mat)
    
    @classmethod
    def from_vector_and_quat(cls, translation_vector, quaternion, position_factor=1, orientation_factor=1):
        mat = R.from_quat(quaternion).as_matrix()
        mat[:3,3] = translation_vector
        return cls(mat, position_factor, orientation_factor)

    def update(self, tensor_or_mat = None, translation_vector =  None, quaternion = None):
        # check if tensor is a torch tensor
        if hasattr(tensor_or_mat, 'cpu'):
            tensor_or_mat = tensor_or_mat.cpu().numpy()
        self.update_from_mat(tensor_or_mat)
    
    def update_from_mat(self, mat):
        self.mat = mat
        self.raw_position = Position(self.mat[:3,3]*self.position_factor*np.array([1,-1,1]))
        self.position = Position(self.filters['position'].apply(self.mat[:3,3])*self.position_factor*np.array([1,-1,1]))
        mat = self.mat[:3,:3]
        self.raw_orientation = Orientation(mat*self.orientation_factor)
        self.orientation = Orientation(self.filters['orientation'].apply(mat)*self.orientation_factor)
    
    def update_from_vector_and_quat(self, translation_vector, quaternion):
        mat = R.from_quat(quaternion).as_matrix()
        mat[:3,3] = translation_vector
        self.update_from_mat(mat)
        
    def __str__(self):
        out = 'position : ' + str(self.position) + ' -- orientation : ' + str(self.orientation)
        return out
    
    def as_list(self):
        return self.position.as_list()+self.orientation.as_list()
    
class Bbox:

    _IMAGE_RESOLUTION = (1280, 720)
    
    @classmethod
    def set_image_resolution(cls, resolution):
        cls._IMAGE_RESOLUTION = resolution
        
    def __init__(self, label, tensor) -> None:
        self.filter = LandmarksSmoothingFilter(min_cutoff=0.001, beta=0.5, derivate_cutoff=1, disable_value_scaling=True)
        self.label = label
        self.color = (255, 0, 0)
        self.thickness = 2
        self.img_resolution = Bbox._IMAGE_RESOLUTION
        self.update_coordinates(tensor)

    def draw(self, img):
        cv2.rectangle(img, self.corner1, self.corner2, self.color, self.thickness)
    
    def update_coordinates(self,tensor):
        box = self.filter.apply(tensor.cpu().numpy())
        self.corner1 = (min(int(box[0]), self.img_resolution[0]-self.thickness), min(int(box[1]), self.img_resolution[1]-self.thickness))
        self.corner2 = (min(int(box[2]), self.img_resolution[0]-self.thickness), min(int(box[3]), self.img_resolution[1]-self.thickness))

    def __str__(self) -> str:
        s= 'Box <'+self.label+'> : ('+str(self.corner1)+','+str(self.corner2)+')'
        return s
    
    def update_display(self, color, thickness):
        self.color = color
        self.thickness = thickness

class RigidObjectState:
    def __init__(self, pose = None, timestamp=None, position_factor=1, orientation_factor=1) -> None:
        self.position_factor = position_factor
        self.orientation_factor = orientation_factor
        if pose is None or timestamp is None:
            self.pose = None
            self.timestamp = None
        else:
            self.pose = Pose(pose, position_factor, orientation_factor)
            self.timestamp = timestamp
    
    @classmethod
    def from_prediction(cls, prediction, timestamp, position_factor=1000):
        return cls(prediction, timestamp, position_factor)
    
    def update(self, pose, timestamp=None):
        if isinstance(pose, Pose): 
            self.pose = pose
        else:
            self.pose = Pose(pose, self.position_factor, self.orientation_factor)
        self.timestamp = timestamp
        
    def propagate(self, timestamp):
        self.timestamp = timestamp
        #TODO: propagate pose for tracking moving objects
    
    def as_list(self, timestamp=True, pose=True):
        repr_list = []
        if timestamp:
            repr_list.append(self.timestamp)
        if pose:
            repr_list += self.pose.as_list()
        return repr_list
        
class GraspingHandState:
    def __init__(self,  position=None, normalized_landmarks=None,  timestamp = None) -> None:
        self.position = Position(position)
        self.normalized_landmarks = normalized_landmarks
        # if landmarks is None:
        #     self.landmarks = np.zeros((21,3))
        # else:
        #     self.landmarks = landmarks
        self.new_position = self.position
        self.new_normalized_landmarks = self.normalized_landmarks
        
        self.velocity = np.array([0,0,0])
        self.scalar_velocity = 0
        self.normed_velocity = np.array([0,0,0])
        self.position_filtered = self.position
        self.velocity_filtered = self.velocity
        self.filter_position, self.filter_velocity = Filter.both('xyz')
        
        if normalized_landmarks is not None:
            self.normalized_landmarks_velocity = np.zeros((21,3))
            self.normalized_landmarks_filtered = self.normalized_landmarks
            self.normalized_landmarks_velocity_filtered = self.normalized_landmarks_velocity            
            self.filter_normalized_landmarks, self.filter_normalized_landmarks_velocity= Filter.both('normalized_landmarks')
            
            
        if timestamp is None:
            self.last_timestamp = time.time()
        else:
            self.last_timestamp = timestamp
        self.was_updated = False
        self.scalar_velocity_threshold = 20 #mm/s
        
    @classmethod
    def from_hand_detection(cls, hand_detection: HandPrediction, timestamp = 0):
        return cls(hand_detection.position, hand_detection.normalized_landmarks, timestamp)

    def update_position(self, position):
        self.new_position = Position(position)
        
    def update_normalized_landmarks(self, normalized_landmarks):
        self.new_normalized_landmarks = normalized_landmarks
    
    def update(self, new_input):
        if isinstance(new_input, HandPrediction):
            self.update_position(new_input.position)
            self.update_normalized_landmarks(new_input.normalized_landmarks)
        elif isinstance(new_input, Position):
            self.update_position(new_input)
        elif new_input is None:
            self.update_normalized_landmarks(new_input)
        else:
            print(f'weird input : {new_input}')
        print(f'updated with {type(new_input)} : {new_input}')
        self.was_updated = True
        
    def propagate(self, timestamp):
        
        elapsed = timestamp - self.last_timestamp
        self.propagate_position(elapsed)
        if self.normalized_landmarks is not None:
            self.propagate_normalized_landmarks(elapsed)
        self.last_timestamp = timestamp
        self.was_updated = False
        
    def propagate_position(self, elapsed):
        if not  self.was_updated:
            next_position = self.position
        else:
            next_position = self.new_position
            next_position_filtered = Position(self.filter_position.apply(next_position.v))
            if elapsed >0:
                self.velocity = (next_position_filtered.v - self.position_filtered.v)/elapsed
            self.position_filtered = next_position_filtered
            
            self.velocity_filtered = self.filter_velocity.apply(self.velocity)*np.array([-1,1,1])
            
            self.scalar_velocity = np.linalg.norm(self.velocity_filtered)
            if self.scalar_velocity != 0:
                if self.scalar_velocity > self.scalar_velocity_threshold:
                    self.normed_velocity = self.velocity_filtered/self.scalar_velocity
                else:
                    self.normed_velocity = self.normed_velocity*98/100 + self.velocity_filtered/self.scalar_velocity*2/100
            else:
                self.normed_velocity = np.array([0,0,0])
                
            self.position = next_position
        
    def propagate_normalized_landmarks(self, elapsed):
        if not  self.was_updated:
            next_normalized_landmarks = self.normalized_landmarks + elapsed*self.normalized_landmarks_velocity
            self.normalized_landmarks_velocity = self.normalized_landmarks_velocity
        else:
            next_normalized_landmarks = self.new_normalized_landmarks
            if elapsed >0:
                self.normalized_landmarks_velocity = (self.new_normalized_landmarks - self.normalized_landmarks)/elapsed
            
        self.normalized_landmarks_filtered = self.filter_normalized_landmarks.apply(next_normalized_landmarks)
        self.normalized_landmarks_velocity_filtered = self.filter_normalized_landmarks_velocity.apply(self.normalized_landmarks_velocity)
            
        self.normalized_landmarks = next_normalized_landmarks
    
    def as_list(self, timestamp=True, position=False, normalized_landmarks=False, velocity=False, normalized_landmarks_velocity=False, filtered_position=False, filtered_velocity=False, filtered_normalized_landmarks=False, filtered_normalized_landmarks_velocity=False, normalized_velocity=False, scalar_velocity=False):
        repr_list = []
        if timestamp:
            repr_list.append(self.last_timestamp)
        if position:
            repr_list += self.position.as_list()
        if normalized_landmarks:
            repr_list += self.normalized_landmarks.flatten().tolist()
        if velocity:
            repr_list += self.velocity.tolist()
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

class Trajectory():
    DATA_KEYS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'v']
    def __init__(self, state = None, headers_list=None, attributes_dict=None, file = None) -> None:
        if file is not None:
            #check if file exists and is a csv file
            if not os.path.isfile(file):
                raise ValueError(f'File {file} does not exist')
            elif not file.endswith('.csv'):
                raise ValueError(f'File {file} is not a csv file')
            else:
                self.data = pd.read_csv(file)
        else:       
            self.data = pd.DataFrame(columns=headers_list)
            self.attributes_dict = attributes_dict
            self.add(state)
        self.current_state = None
        self.current_line_index = 0
    
    def add(self, new_data):
        if new_data is not None:
            self.data.loc[len(self.data)] = new_data.as_list(**self.attributes_dict)
            
    def get_data(self):
        return self.data
    
class GraspingHandTrajectory(Trajectory):
    DEFAULT_DATA_KEYS = [ 'Timestamps', 'x', 'y', 'z']
    DEFAULT_ATTRIBUTES  = dict(timestamp=True, filtered_position=True)
    def __init__(self, state = None, headers_list = DEFAULT_DATA_KEYS, attributes_dict=DEFAULT_ATTRIBUTES, file = None) -> None:
        super().__init__(state, headers_list, attributes_dict, file)
        
    @classmethod
    def from_file(cls, file):
        return Trajectory(file=file)
    
    @classmethod
    def from_state(cls, state):
        return cls(state)
    
    def next_position(self):
        if self.current_line_index < len(self.data):
            row = self.data.iloc[self.current_line_index]
            self.current_line_index+=1
            return Position(np.array([row['x'], row['y'], row['z']]), display='cm', swap_y=True)
        

class RigidObjectTrajectory(Trajectory):
    
    DEFAULT_DATA_KEYS = [ 'Timestamps', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    DEFAULT_ATTRIBUTES  = dict(timestamp=True, pose=True)
    def __init__(self, state = None, headers_list = DEFAULT_DATA_KEYS, attributes_dict=DEFAULT_ATTRIBUTES, file = None) -> None:
        super().__init__(state, headers_list, attributes_dict, file)
        
    @classmethod
    def from_file(cls, file):
        return Trajectory(file=file)
    
    @classmethod
    def from_state(cls, state):
        return cls(state)

    def next_pose(self):
        if self.current_line_index < len(self.data):
            row = self.data.iloc[self.current_line_index]
            self.current_line_index+=1
            return Pose.from_vector_and_quat(np.array([row['x'], row['y'], row['z']]), np.array([row['qx'], row['qy'], row['qz'], row['qw']]))
    
class DataWindow:
    def __init__(self, size:int) -> None:
        self.size = size # nb iterations or time limit ?
        self.data = list()
        self.nb_samples = 0
    
    def queue(self, new_data):
        self.data.append(new_data)
        if len(self.data)>= self.size:
            del self.data[0]
        else:
            self.nb_samples+=1

class Filter(LandmarksSmoothingFilter):
    
    
    def __init__(self, key, type='natural') -> None:
        min_cutoffs = {'landmarks' : 0.001, 'world_landmarks' : 0.001, 'xyz': 1, 'normalized_landmarks' : 0.15} 
        betas = {'landmarks' : 0.5, 'world_landmarks' : 0.5, 'xyz': 0.5, 'normalized_landmarks' : 50}
        derivate_cutoff = {'landmarks' : 1, 'world_landmarks' : 1, 'xyz': 0.00000001, 'normalized_landmarks' : 10}

        derivative_min_cutoffs = {'landmarks' : 0.001, 'world_landmarks' : 0.001, 'xyz': 0.1, 'normalized_landmarks' : 0.001}
        derivative_betas = {'landmarks' : 1.5, 'world_landmarks' : 0.5, 'xyz': 0.08, 'normalized_landmarks' : 0.5}
        derivative_derivate_cutoff = {'landmarks' : 1, 'world_landmarks' : 1, 'xyz': 0.1, 'normalized_landmarks' : 1}

        refinable_keys = ['landmarks', 'world_landmarks', 'xyz', 'normalized_landmarks']
        types = ['natural', 'derivative']
        if key not in refinable_keys:
            raise ValueError('key must be in '+str(refinable_keys))
        if type not in types:
            raise  ValueError('type must be in '+str(types))
        if type == 'natural':
            super().__init__(min_cutoff=min_cutoffs[key], beta=betas[key], derivate_cutoff=derivate_cutoff[key], disable_value_scaling=True)
        else:
            super().__init__(min_cutoff=derivative_min_cutoffs[key], beta=derivative_betas[key], derivate_cutoff=derivative_derivate_cutoff[key], disable_value_scaling=True)

    @classmethod
    def natural(cls, key):
        return cls(key, type='natural')
    
    @classmethod
    def derivative(cls, key):
        return cls(key, type='derivative')
    
    @classmethod
    def both(cls, key):
        return cls(key, type='natural'), cls(key, type='derivative')
    
class HandConeImpactsWindow(DataWindow):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.nb_impacts = 0

    # def mean(self):
    #     if self.nb_samples==0:
    #         return None
    #     sum = 0
    #     for i in range(self.data):
    #         sum+=np.mean(self.data[i])
    #     return sum/self.nb_samples

    def queue(self, new_data):
        self.data.append(new_data)
        #print('new_data', type(new_data))
        if len(self.data)>= self.size:
            deleted =  self.data.pop(0)
            self.nb_impacts = self.nb_impacts - len(deleted) + len(new_data)
        else:
            self.nb_impacts = self.nb_impacts + len(new_data)
            self.nb_samples+=1
    def mean(self):
        sum = np.array([0.,0.,0.])
        for l_imp in self.data:
            if len(l_imp)>0:
                #print('l_imp', l_imp)
                a=np.array([np.mean(l_imp[:,i]) for i in range(3)])
                #print('a',a)
                sum += a
        nb = len(self.data)
        if nb >0:
            mean = sum / nb
        else:
            mean = sum
        return mean
    def get_nb_impacts(self):
        return self.nb_impacts
    
class TargetDetector:
    def __init__(self, hand_label, window_size = 100) -> None:
        self.window_size = window_size
        self.potential_targets = {}
        self.hand_label = hand_label
        # self.derivative_min_cutoff = derivative_min_cutoff
        # self.derivative_beta = derivative_beta
        # self.derivative_derivate_cutoff = derivative_derivate_cutoff
    
    def new_impacts(self, obj, impacts, relative_hand_pos, elapsed):
        label = obj.label
        obj_pose = obj.mesh_transform
        relative_hand_pos = Position(relative_hand_pos)
        #print('impacts', impacts)
        if label in self.potential_targets:
            self.potential_targets[label].update(impacts, relative_hand_pos, elapsed) 
        elif len(impacts)>0:
            self.potential_targets[label] = Target(obj, impacts, relative_hand_pos)

    def get_most_probable_target(self):
        if self.potential_targets:
            n_impacts = {}
            n_tot = 0
            to_del_keys=[]
            for lab, target in self.potential_targets.items():
                n_impacts[lab] = target.projected_collison_window.get_nb_impacts()
                if n_impacts[lab]<=0:
                    to_del_keys.append(lab)
                n_tot +=n_impacts[lab]

            for key in to_del_keys:
                del self.potential_targets[key]

            if n_tot == 0:
                most_probable_target =  None                
            else:
                for lab in self.potential_targets:
                    self.potential_targets[lab].set_impact_ratio(n_impacts[lab]/n_tot)
                max_ratio_label = max(n_impacts, key = n_impacts.get)
                most_probable_target = self.potential_targets[max_ratio_label]
        else:
            most_probable_target =  None
        
        # print(self.hand_label,' most_probable_target : ',most_probable_target)
        return most_probable_target, self.potential_targets

       #def set_hand_pos_vel(self, hand_pos, hand_vel):
        #self.hand_pos = hand_pos
        #self.hand_vel = hand_vel

    #def compute_time_before_impact(self):
    #    for tar in self.potential_targets:
    #         """

class Target:
    def __init__(self, obj, impacts,  relative_hand_pos,  window_size = 10) -> None:
        self.object = obj
        self.label = obj.label
        self.window_size = window_size
        self.projected_collison_window = HandConeImpactsWindow(window_size)
        print('POTENTIAL TARGET ')
        self.ratio=0
        self.last_update_time = time.time()
        self.distance_derivative = 0
        self.projected_collison_window.queue(impacts)
        self.predicted_impact_zone = Position(self.projected_collison_window.mean())
        self.distance_to_hand = Position.distance(relative_hand_pos, self.predicted_impact_zone)
        self.time_before_impact = 0
        self.grip = 'None'
        # self.derived_filter = LandmarksSmoothingFilter(min_cutoff=derivative_min_cutoff, beta=derivative_beta, derivate_cutoff=derivative_derivate_cutoff, disable_value_scaling=True)
        self.relative_hand_vel = np.array([0,0,0])
        self.relative_hand_pos = relative_hand_pos

    def update(self, impacts, new_relative_hand_pos,elapsed):        
        self.relative_hand_vel = (new_relative_hand_pos.v - self.relative_hand_pos.v)/elapsed
        self.relative_hand_pos = new_relative_hand_pos
        self.projected_collison_window.queue(impacts)
        self.predicted_impact_zone = Position(self.projected_collison_window.mean())
        self.compute_time_before_impact(new_relative_hand_pos)
        self.find_grip()
    
    def set_impact_ratio(self, ratio):
        self.ratio = ratio

    def compute_time_before_impact(self, relative_hand_pos):
        new_distance_to_hand = Position.distance(relative_hand_pos, self.predicted_impact_zone)
        print('hand_pos', relative_hand_pos)
        print('self.predicted_impact_zone)', self.predicted_impact_zone)
        t = time.time()
        dt = t-self.last_update_time
        self.last_update_time = t
        if dt != 0 :
            distance_der = (self.distance_to_hand - new_distance_to_hand)/dt
            if distance_der !=0:
                #self.time_before_impact = new_distance_to_hand
                self.time_before_impact = new_distance_to_hand/distance_der
            self.distance_to_hand = new_distance_to_hand
            self.distance_derivative = distance_der
        #print('new_distance_to_hand',new_distance_to_hand)
        print('time to impact', int(self.time_before_impact ), 'ms')

    def __str__(self) -> str:
        out = 'Target: '+self.object.label + ' - nb impacts: ' + str(self.projected_collison_window.nb_impacts) + ' - ratio: ' + str(self.ratio)
        return out
    
    def get_proba(self):
        return self.ratio

    def get_time_of_impact(self, unit = 'ms'):
        if unit == 'ms':
            return int(self.time_before_impact*1000)
        if unit == 's':
            return int(self.time_before_impact)
    
    def find_grip(self):
        if abs(self.predicted_impact_zone.v[0])>20:
            self.grip = 'PINCH'
        else:
            self.grip = 'PALMAR'

    def get_grip(self):
        return self.grip

class GripSelector:
    def __init__(self) -> None:
        pass

def kill_gpu_processes():
    # use the command nvidia-smi and then grep "grasp_int" and "python" to get the list of processes running on the gpu
    # execute the command in a subprocess and get the output
    try:
        processes = subprocess.check_output("nvidia-smi | grep 'i_grip' | grep 'python'", shell=True)
        # split the output into lines
        processes = processes.splitlines()
        # get rid of the b' at the beginning of each line
        processes = [str(process)[2:] for process in processes]
        ids=[]
        # loop over the lines
        for process in processes:
            # split the line into words and get the fifth word, which is the process id
            id = process.split()[4]
            ids.append(id)
            # kill the process
            kill_command = f"sudo kill -9 {id}"
            subprocess.call(kill_command, shell=True)
        print(f"Killed processes with ids {ids}")
    except Exception as e:
        print(f"No remnant processes found on the gpu")