#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import cv2
import subprocess
import pandas as pd
import os
# from i_grip.HandDetectors2 import HandPrediction
from i_grip.Filters import LandmarksSmoothingFilter
from findiff import FinDiff
# import pynumdiff
from scipy.interpolate import CubicSpline
import trimesh as tm

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
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @staticmethod
    def distance(p1 : 'Position', p2 : 'Position'):
        if p1 is None :
            print('p1 is None')
        if p2 is None :
            print('p2 is None')
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
    def __init__(self, pose_tensor_or_mat, position_factor=1, orientation_factor=1, filtered=False , flip_pos_y = False) -> None:
        self.filtered = filtered
        self.position_factor = position_factor
        self.orientation_factor = orientation_factor
        self.flip_pos_y = flip_pos_y
        if filtered:
            self.filter_position, self.filter_velocity = Filter.both('position')
            self.filter_orientation, self.filter_angular_velocity = Filter.both('orientation')
        self.update_from_mat = self.update_from_mat_filtered if filtered else self.update_from_mat_raw
        
        if hasattr(pose_tensor_or_mat, 'cpu'):
            pose_tensor_or_mat = pose_tensor_or_mat.cpu().numpy()
            
        if isinstance(pose_tensor_or_mat, Pose):
            self.mat = pose_tensor_or_mat.mat
            self.position = pose_tensor_or_mat.position
            self.orientation = pose_tensor_or_mat.orientation
            # self.filtered = pose_tensor_or_mat.filtered
            # self.position_factor = pose_tensor_or_mat.position_factor
            # self.orientation_factor = pose_tensor_or_mat.orientation_factor
            # self.filter_position = pose_tensor_or_mat.filter_position
            # self.filter_velocity = pose_tensor_or_mat.filter_velocity
            # self.filter_orientation = pose_tensor_or_mat.filter_orientation
            # self.filter_angular_velocity = pose_tensor_or_mat.filter_angular_velocity
            # self.update_from_mat = pose_tensor_or_mat.update_from_mat
        elif pose_tensor_or_mat is None:
            self.mat = np.zeros((4,4))
            self.mat[3,3] = 1
            self.position = Position(np.zeros(3))
            self.orientation = Orientation(np.zeros(3))
            self.update(pose_tensor_or_mat, flip_pos_y=self.flip_pos_y)
        elif isinstance(pose_tensor_or_mat, np.ndarray):
            if pose_tensor_or_mat.shape != (4,4):
                raise ValueError('Pose must be initialized with a 4x4 matrix')
            self.update_from_mat(pose_tensor_or_mat,flip_pos_y=self.flip_pos_y)
        else:
            raise TypeError('Pose must be initialized with a 4x4 matrix or a Pose object')
    
    @classmethod
    def from_vector_and_quat(cls, translation_vector, quaternion, position_factor=1, orientation_factor=1, filtered=False, flip_pos_y = False):
        print('translation_vector', translation_vector)
        print('quaternion', quaternion)
        mat = np.zeros((4,4))
        mat[3,3] = 1
        mat[:3,:3] = R.from_quat(quaternion).as_matrix()
        mat[:3,3] = translation_vector
        return cls(mat, position_factor, orientation_factor, filtered, flip_pos_y)

    def update(self, tensor_or_mat = None, translation_vector =  None, quaternion = None, flip_pos_y = False):
        # check if tensor is a torch tensor
        if tensor_or_mat is None:
            if translation_vector is not None and quaternion is not None:
                self.update_from_vector_and_quat(translation_vector, quaternion, flip_pos_y = flip_pos_y)
            else:
                raise ValueError('Pose must be updated with a 4x4 matrix, a Pose object or a translation vector and a quaternion')
        elif isinstance(tensor_or_mat, Pose):
            self.update_from_mat(tensor_or_mat.mat, flip_pos_y = flip_pos_y)
        elif hasattr(tensor_or_mat, 'cpu'):
            tensor_or_mat = tensor_or_mat.cpu().numpy()
            self.update_from_mat(tensor_or_mat, flip_pos_y = flip_pos_y)
        else:
            raise TypeError('Pose must be updated with a 4x4 matrix, a Pose object or a translation vector and a quaternion')
    
    def update_from_mat_raw(self, mat, flip_pos_y = False):
        self.mat = mat
        if flip_pos_y:
            self.position = Position(self.mat[:3,3]*self.position_factor*np.array([1,-1,1]))
        else:
            self.position = Position(self.mat[:3,3]*self.position_factor)
        rot_mat = self.mat[:3,:3]
        self.orientation = Orientation(rot_mat*self.orientation_factor)
        
    def update_from_mat_filtered(self, mat, flip_pos_y = False):
        self.mat = mat
        if flip_pos_y:
            self.position = Position(self.filter_position.apply(self.mat[:3,3])*self.position_factor*np.array([1,-1,1]))
        else:
            self.position = Position(self.filter_position.apply(self.mat[:3,3])*self.position_factor)
        rot_mat = self.mat[:3,:3]
        self.orientation = Orientation(self.filter_orientation.apply(rot_mat)*self.orientation_factor)
    
    def update_from_vector_and_quat(self, translation_vector, quaternion, flip_pos_y = False):
        mat = np.zeros((4,4))
        mat[3,3] = 1
        mat[:3,:3] = R.from_quat(quaternion).as_matrix()
        if flip_pos_y:
            mat[:3,3] = translation_vector*np.array([1,-1,1])
        else:
            mat[:3,3] = translation_vector
        self.update_from_mat(mat)
        
    def __str__(self):
        out = 'position : ' + str(self.position) + ' -- orientation : ' + str(self.orientation)
        return out
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def as_list(self):
        return self.position.as_list()+self.orientation.as_list()
    
class Bbox:

    _IMAGE_RESOLUTION = (1280, 720)
    _IMAGE_RESOLUTION = [640., 480.]
    
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
        

class Filter(LandmarksSmoothingFilter):    
    def __init__(self, key, type='natural') -> None:
        min_cutoffs = {'landmarks' : 0.001, 'world_landmarks' : 0.001, 'position': 1, 'normalized_landmarks' : 0.15, 'orientation' : 0.0001} 
        betas = {'landmarks' : 0.5, 'world_landmarks' : 0.5, 'position': 0.5, 'normalized_landmarks' : 50, 'orientation' : 0.5}
        derivate_cutoff = {'landmarks' : 1, 'world_landmarks' : 1, 'position': 0.00000001, 'normalized_landmarks' : 10, 'orientation' : 1}

        derivative_min_cutoffs = {'landmarks' : 0.001, 'world_landmarks' : 0.001, 'position': 0.1, 'normalized_landmarks' : 0.001, 'orientation' : 0.0001}
        derivative_betas = {'landmarks' : 1.5, 'world_landmarks' : 0.5, 'position': 0.08, 'normalized_landmarks' : 0.5, 'orientation' : 0.5}
        derivative_derivate_cutoff = {'landmarks' : 1, 'world_landmarks' : 1, 'position': 0.1, 'normalized_landmarks' : 1, 'orientation' : 1}

        refinable_keys = ['landmarks', 'world_landmarks', 'position', 'normalized_landmarks', 'orientation']
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
        self.mesh_updated = False
        self.label = None

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
    
    def was_mesh_updated(self):
        return self.mesh_updated

    def set_mesh_updated(self, bool):
        self.mesh_updated = bool
        # print(f'{self.label} mesh updated : {self.mesh_updated}')
    
    def update_trajectory(self):
        # add a new line to the trajectory dataframe with the current timestamp position and velocity
        self.state.trajectory.add(self.state)
    
    def get_trajectory(self):
        return self.state.trajectory.get_data()
    
    def load_trajectory(self, trajectory):
        self.state.trajectory = trajectory
        
    def get_mesh_transform(self):
        return self.mesh_transform
    
    def get_inv_mesh_transform(self):
        return self.inv_mesh_transform
  
   

class State:
    def __init__(self) -> None:
        self.updated = False
        self.propagated = False
        self.last_timestamp = None
        
    def set_updated(self, updated):
        self.updated = updated
        
    def was_updated(self):
        return self.updated
    
    def set_propagated(self, propagated):
        self.propagated = propagated
    
    def was_propagated(self):
        return self.propagated
    
    def get_timestamp(self):
        return self.last_timestamp

class Trajectory():
    
    DATA_KEYS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'v']
    
    def __init__(self, state = None, headers_list=DATA_KEYS, attributes_dict=None, file = None, dataframe =None, limit_size = None) -> None:
        self.attributes_dict = attributes_dict
        self.limit_size = limit_size
        if dataframe is not None:
            self.data = dataframe                
        elif file is not None:
            #check if file exists and is a csv file
            if not os.path.isfile(file):
                raise ValueError(f'File {file} does not exist')
            elif not file.endswith('.csv'):
                raise ValueError(f'File {file} is not a csv file')
            else:
                self.data = pd.read_csv(file)
        else:       
            self.data = pd.DataFrame(columns=headers_list)
            self.add(state)
        self.current_state = None
        self.current_line_index = 0
        # self.states = []
        
    @classmethod
    def from_dataframe(cls, df:pd.DataFrame):
        return cls(dataframe=df)
    
    @classmethod
    def from_file(cls, file):
        return cls(file=file)
    
    @classmethod
    def from_state(cls, state, limit_size = None):
        return cls(state = state, limit_size = limit_size)
    
    def add(self, new_state:State, extrapolated=False):
        if new_state is not None:
            timestamp = new_state.get_timestamp()
            if timestamp in self.data['Timestamps'].values:
                print(f'Timestamp {timestamp} already in trajectory, ignoring new state')
                return
            else:
                new_entries = new_state.as_list(**self.attributes_dict)+[extrapolated]
                self.data.loc[len(self.data)] = new_entries
        if self.limit_size is not None:
            if len(self.data)>self.limit_size:
                #delete the first line
                self.data = self.data.iloc[1:,:]
                #reset the index
                self.data.reset_index(drop=True, inplace=True)
            # self.states.append(new_state)
            
    def get_data(self):
        return self.data

    def __iter__(self):
        self.current_line_index = 0
        return self
    
    def __next__(self):
        if self.current_line_index < len(self.data):
            row = self.data.iloc[self.current_line_index]
            self.current_line_index+=1
            return row
        else:
            raise StopIteration
    
    def find_plane(self, nb_points):
        # find regression plane for the last nb_points of the trajectory
        if len(self.data) < nb_points:
            raise ValueError('Not enough data points to find a plane')
        else:
            last_points = self.data.iloc[-nb_points:]
            x = last_points['x']
            y = last_points['y']
            z = last_points['z']
            A = np.array([x,y,np.ones(len(x))]).T
            B = z
            a,b,c = np.linalg.lstsq(A,B,rcond=None)[0]
            return a,b,c
    def __repr__(self) -> str:
        return self.data.__repr__()
        

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
        
def findiff_diff(times, values, diff_order=1, diff_acc=10):
    # times and values must be numpy arrays
    # times must be sorted
    # values must be a 1D array
    # times and values must have the same length
    # times must be in seconds
    # values must be in mm
    if len(times)!=len(values):
        raise ValueError('times and values must have the same length')
    if len(times.shape)!=1:
        raise ValueError('times must be a 1D array')
    if not np.all(np.diff(times)>=0):
        raise ValueError('times must be sorted')
    if not np.all(np.diff(times)>0):
        print('Warning : times must be strictly increasing')
    # print('order', diff_order)
    print('times', times- times[0])
    # dt = np.gradient(times)
    print(f'diff_acc : {diff_acc}')
    d_dt = FinDiff(0, times, diff_order, acc=diff_acc)
    dvals = np.gradient(values, times)
    # print('dt', dt[-5:])
    # print('values', values[-5:])
    d_values = d_dt(values)
    # print('d_values', d_values[-5:])
    # d_values2 = np.gradient(values[:,0], times)
    # print('d_values2', d_values2[-5:])
    # dt3 = np.array((dt.T, dt.T, dt.T))
    # d_values2_dt = d_values2/dt3
    # print('d_values2_dt', d_values2_dt[-5:])
    
    # x = np.linspace(0,10,11)
    # dx = x[1]-x[0]
    # f =2*x
    # print('x', x)
    # print('dx', dx)
    # print('f', f)   
    # d_dx = FinDiff(0, dx, diff_order, acc=diff_acc)
    # vx = d_dx(f)
    # print('vx', vx)
    
    return d_values[-1]
    # vals = d_dx(f)
    # return vals[-1]

def pynumdiff_diff(times, values, diff_order=1, diff_acc=5):
    # times and values must be numpy arrays
    # times must be sorted
    # values must be a 1D array
    # times and values must have the same length
    # times must be in seconds
    # values must be in mm
    if len(times)!=len(values):
        raise ValueError('times and values must have the same length')
    if len(times.shape)!=1:
        raise ValueError('times must be a 1D array')
    if len(values.shape)!=1:
        raise ValueError('values must be a 1D array')
    if times.shape != values.shape:
        raise ValueError('times and values must have the same shape')
    if not np.all(np.diff(times)>=0):
        raise ValueError('times must be sorted')
    if not np.all(np.diff(times)>0):
        print('Warning : times must be strictly increasing')
    dt = np.diff(times)
    d_dt = pynumdiff.total_variation_regularization.d_dt(dt, diff_order, acc=diff_acc)
    last_vs = pynumdiff.smooth_finite_difference.meandiff
    d_values = d_dt(values)
    return d_values[-1]

def my_polydiff(timestamps, values, poly_degree, diff_order=1, diff_acc=None):
    # times and values must be numpy arrays
    # times must be sorted
    # values must be a 1D array
    # times and values must have the same length
    # times must be in seconds
    # values must be in mm
    if len(timestamps)!=len(values):
        raise ValueError('times and values must have the same length')
    if len(timestamps.shape)!=1:
        raise ValueError('times must be a 1D array')
    if len(values.shape)!=3:
        raise ValueError('values must be a 3D array')
    if not np.all(np.diff(timestamps)>=0):
        raise ValueError('times must be sorted')
    if not np.all(np.diff(timestamps)>0):
        print('Warning : times must be strictly increasing')
    
    if diff_acc is not None:
        timestamps = timestamps[::diff_acc]

    # Example 3D variable time series (replace this with your data)
    x_values = values[:,0]
    y_values = values[:,1]
    z_values = values[:,2]
    
    # Fit a polynomial function to the data (degree can be adjusted)
    poly_degree = 2  # Change the degree of the polynomial as needed
    poly_x = np.polynomial.polynomial.Polynomial.fit(timestamps, x_values, poly_degree)
    poly_y = np.polynomial.polynomial.Polynomial.fit(timestamps, y_values, poly_degree)
    poly_z = np.polynomial.polynomial.Polynomial.fit(timestamps, z_values, poly_degree)

    coeff_x = poly_x.coef
    coeff_y = poly_y.coef
    coeff_z = poly_z.coef

    # Generate a polynomial function using the coefficients
    
    poly_func_x = np.poly1d(coeff_x)
    poly_func_y = np.poly1d(coeff_y)
    poly_func_z = np.poly1d(coeff_z)

    # Calculate derivatives of the polynomial functions (velocity)
    delta_t = np.diff(timestamps)  # Calculate time differences
    vel_poly_x = np.polyder(poly_func_x, m=diff_order)  # Derivative for velocity in x
    vel_poly_y = np.polyder(poly_func_y, m=diff_order)  # Derivative for velocity in y
    vel_poly_z = np.polyder(poly_func_z, m=diff_order)  # Derivative for velocity in z

    # Calculate velocities at specific timestamps
    velocities_x = vel_poly_x(timestamps[-1]) 
    velocities_y = vel_poly_y(timestamps[-1]) 
    velocities_z = vel_poly_z(timestamps[-1]) 
    # velocities_x = vel_poly_x(timestamps[-1]) / delta_t
    # velocities_y = vel_poly_y(timestamps[-1]) / delta_t
    # velocities_z = vel_poly_z(timestamps[-1]) / delta_t

    # Print velocities at specific timestamps
    print("Velocity in x-direction:", velocities_x)
    print("Velocity in y-direction:", velocities_y)
    print("Velocity in z-direction:", velocities_z)
    

    return d_values[-1]


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
