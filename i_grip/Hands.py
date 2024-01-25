import numpy as np
import pandas as pd
import trimesh as tm
import cv2
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from i_grip.utils2 import *
from i_grip import HandDetectors2 as hd
from i_grip.Targets import TargetDetector
from i_grip.Objects import RigidObject


    
class GraspingHandTrajectory(Trajectory):
    
    DEFAULT_DATA_KEYS = [ 'Timestamps', 'x', 'y', 'z', 'Extrapolated']
    DEFAULT_ATTRIBUTES  = dict(timestamp=True, filtered_position=True)
    
    def __init__(self, state = None, headers_list = DEFAULT_DATA_KEYS, attributes_dict=DEFAULT_ATTRIBUTES, file = None, dataframe=None, fit_method = 'np_poly') -> None:
        super().__init__(state, headers_list, attributes_dict, file, dataframe)
        self.poly_coeffs = {}
        self.polynomial_function = None
        self.was_fitted = False
        if fit_method == 'np_poly':
            self.fit = self.polynomial_fit
            self.extrapolate = self.np_poly_extrapolate
        elif fit_method == 'skl_spline':
            self.model = make_pipeline(SplineTransformer(n_knots=2, degree=2, extrapolation='continue'), Ridge(alpha=1e-3))
            self.fit = self.skl_fit
            self.extrapolate = self.skl_extrapolate
        elif fit_method == 'skl_poly':
            self.model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-3))
            self.fit = self.skl_fit
            self.extrapolate = self.skl_extrapolate
    
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
    
    def skl_fit(self, nb_points):
        # find polynomial fit for the last nb_points of the trajectory
        if len(self.data) < nb_points:
            print('Not enough data points to find a polynomial fit')
        else:
            last_points = self.data.iloc[-nb_points:]
            t = last_points['Timestamps'].values.reshape(-1,1)
            xyz = last_points[['x','y','z']].values 
            print(f'xyz : {xyz}')
            print(f't : {t}')
            self.model.fit(t, xyz)
            self.was_fitted = True
    
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
            x = self.poly_coeffs['x'](timestamps)
            y = self.poly_coeffs['y'](timestamps)
            z = self.poly_coeffs['z'](timestamps)
        return np.array([x,y,z])

    def skl_extrapolate(self, timestamps): 
        if not self.was_fitted:
            # print('No model found, please use skl_fit() first')
            x = self.data.loc[len(self.data)-1]['x']
            y = self.data.loc[len(self.data)-1]['y']
            z = self.data.loc[len(self.data)-1]['z']
            return np.array([x,y,z])
        else:
            return self.model.predict(np.array([[timestamps]]))[0]
    
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
    
    def __init__(self, input, label = None, timestamp=None,  compute_velocity_cone = False,plotter = None)-> None:
        super().__init__()
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
                self.state = GraspingHandState.from_dataframe(input)
            else:   
                self.state = None
        
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
        self.impact_locations = {}
        self.hand_pos_obj_frame = {}
        self.impact_locations_list = {}
        self.rays_vizualize_list = {}
        self.define_mesh_representation()
        self.define_velocity_cone()
        self.update_mesh()
        
        self.target_detector = TargetDetector(self.label, self.plot_color, plotter=self.plotter)
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
    
    def define_velocity_cone(self,cone_max_length = 500, cone_min_length = 50, vmin=30, vmax=200, cone_max_diam = 200,cone_min_diam = 50, n_layers=2):
        self.cone_angle = np.pi/8
        self.n_layers = n_layers
        self.cone_lenghts_spline = spline(vmin,vmax, cone_min_length, cone_max_length)
        # self.cone_diam_spline = spline(vmin,vmax, cone_min_diam, cone_max_diam)
        self.cone_diam_spline = spline(vmin,vmax, cone_max_diam, cone_min_diam)
    
    def update2(self, detected_hand):
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
        
    def propagate2(self, timestamp=None):
        # print(f'timestamp : {timestamp}')
        self.state.propagate_all(timestamp)
        # self.state.propagate(timestamp)
        self.set_mesh_updated(False)
        # self.update_trajectory()
        # self.update_meshes()
              
    def update_mesh(self):
        if  self.was_mesh_updated():
            return
        self.mesh_position = Position(self.state.position_filtered*np.array([-1,1,1]))
        # self.mesh_position = Position(self.state.velocity_filtered*np.array([-1,1,1])+np.array([0,0,500]))
        self.mesh_transform= tm.transformations.compose_matrix(translate = self.mesh_position.v)
        if self.compute_velocity_cone:
            self.make_rays_from_trajectory()
        self.set_mesh_updated(True)
        
    def make_rays(self): 
        vdir = self.state.get_movement_direction()*np.array([-1,1,1])
        svel = self.state.scalar_velocity
        if True:
        # if svel > 10:
            cone_len = self.cone_lenghts_spline(svel)
            cone_diam = self.cone_diam_spline(svel)
            random_vector = np.random.rand(len(vdir))

            # projection
            projection = np.dot(random_vector, vdir)

            # vecteur orthogonal
            orthogonal_vector = random_vector - projection * vdir

            # vecteur unitaire orthogonal
            orthogonal_unit_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
            orthogonal_unit_vector2 = np.cross(vdir, orthogonal_unit_vector)

            ray_directions_list = [ vdir*cone_len + (-cone_diam/2+i*cone_diam/(self.n_layers-1)) * orthogonal_unit_vector+ (-cone_diam/2+j*cone_diam/(self.n_layers-1)) * orthogonal_unit_vector2 for i in range(self.n_layers) for j in range(self.n_layers) ] 

            self.ray_origins = np.vstack([self.mesh_position.v for i in range(self.n_layers) for j in range(self.n_layers) ])
            self.ray_directions = np.vstack(ray_directions_list)
            self.ray_visualize = tm.load_path(np.hstack((
                self.ray_origins,
                self.ray_origins + self.ray_directions)).reshape(-1, 2, 3))     
        else:
            self.ray_visualize = None
    
    def make_rays_from_trajectory(self):
        self.get_future_trajectory_points()
        vdir = self.state.get_movement_direction()*np.array([-1,1,1])
        svel = self.state.scalar_velocity
        self.impact_locations_list = {}
        ray_origins_list = []
        ray_directions_list = []
        i=0
        for point in self.future_points:
        # for point in self.future_points[:-1]:
            weight = 1/(i+1)
            if i==0:
                prev_point = point
            else:
                vdir = (point - prev_point)
                # next_point= self.future_points[i+1]
                # vdir = (next_point - point)
                svel = np.linalg.norm(vdir)
                vdir = vdir/svel
                svel = svel/0.01
            ray_origins, ray_directions = self.get_rays_from_point(point, vdir, svel)
            prev_point = point
            ray_origins_list.append(ray_origins)
            ray_directions_list.append(ray_directions)
            i+=1
        if len(ray_origins_list)>0:
            self.ray_origins = np.vstack(ray_origins_list)
            self.ray_directions = np.vstack(ray_directions_list)
            self.ray_visualize =  tm.load_path(np.hstack((
                self.ray_origins,
                self.ray_origins + self.ray_directions)).reshape(-1, 2, 3))    
    
    def get_rays_from_point(self, point, vdir, svel):
        cone_len = self.cone_lenghts_spline(svel)
        cone_diam = self.cone_diam_spline(svel)
        random_vector = np.random.rand(len(vdir))

        # projection
        projection = np.dot(random_vector, vdir)

        # vecteur orthogonal
        orthogonal_vector = random_vector - projection * vdir

        # vecteur unitaire orthogonal
        orthogonal_unit_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
        orthogonal_unit_vector2 = np.cross(vdir, orthogonal_unit_vector)

        ray_directions_list = [ vdir*cone_len + (-cone_diam/2+i*cone_diam/(self.n_layers-1)) * orthogonal_unit_vector+ (-cone_diam/2+j*cone_diam/(self.n_layers-1)) * orthogonal_unit_vector2 for i in range(self.n_layers) for j in range(self.n_layers) ] 

        ray_origins = np.vstack([point for i in range(self.n_layers) for j in range(self.n_layers) ])
        ray_directions = np.vstack(ray_directions_list)
        return ray_origins, ray_directions
    
    def check_target(self, obj:RigidObject, mesh, mode='predicted_traj'):
        if not (self.was_mesh_updated() or obj.was_mesh_updated()):
            # print(f'no need to check target for {obj.label}')
            new = False
            self.impact_locations[obj.label] = []
        else:
            # print(f'checking target for {obj.label}')
            new = True
            
            self.target_detector.poke_target(obj)
            
            inv_trans = obj.inv_mesh_transform
            self.hand_pos_obj_frame[obj.label] = (inv_trans@self.mesh_position.ve)[:3]
            
            self.get_distance_to(obj)
            
            ray_origins_obj_frame = []
            self.impact_locations[obj.label] = []
            for point in self.future_points:
                #expand  point to a 4d vector adding 1 as the last component
                point = np.append(point, 1)
                pos_obj_frame = (inv_trans@point)[:3]
                ray_origins_obj_frame +=[ pos_obj_frame for i in range(self.n_layers) for j in range(self.n_layers) ]
            ray_origins_obj_frame = np.vstack(ray_origins_obj_frame)
            
            rot = inv_trans[:3,:3]
            ray_directions_obj_frame = np.vstack([rot @ ray_dir for ray_dir in self.ray_directions])    
            try:
                self.impact_locations[obj.label], _, _ = mesh.ray.intersects_location(ray_origins=ray_origins_obj_frame,
                                                                                ray_directions=ray_directions_obj_frame,
                                                                                multiple_hits=False)
            except:
                self.impact_locations[obj.label] = []
            self.target_detector.new_impacts(obj, self.impact_locations[obj.label], self.hand_pos_obj_frame[obj.label], self.elapsed)
        # obj.distance_to(self)
        return new, self.impact_locations[obj.label]
    
    def check_target3(self, obj:RigidObject, mesh, mode='predicted_traj'):
        if self.was_mesh_updated() or obj.was_mesh_updated():
            print(f'checking target for {obj.label}')
            new = True
            self.target_detector.poke_target(obj)
            self.hand_pos_obj_frame[obj.label] = (obj.inv_mesh_transform@self.mesh_position.ve)[:3]
            if mode == 'predicted_traj':
                self.check_target_from_trajectory(obj.label, obj.inv_mesh_transform, mesh)
            else:
                self.check_target_from_state(obj.label, obj.inv_mesh_transform, mesh)
        else:
            print(f'no need to check target for {obj.label}')
            new = False
            impact_locations = []
        self.target_detector.update_distance_to(obj, self.impact_locations[obj.label], self.elapsed)
        self.target_detector.new_impacts(obj, self.impact_locations[obj.label], self.hand_pos_obj_frame[obj.label], self.elapsed)
        # obj.distance_to(self)
        return new, impact_locations

    def check_target2(self, obj:RigidObject, mesh):
        if self.was_mesh_updated() or obj.was_mesh_updated():
            inv_trans = obj.inv_mesh_transform
            self.hand_pos_obj_frame[obj.label] = (inv_trans@self.mesh_position.ve)[:3]
            # ray_directions_list = [ np.array([0,-1000,0]) + np.array([-200+i*100,0,-200+j*100] ) for i in range(5) for j in range(5) ]
            rot = inv_trans[:3,:3]
            ray_origins_obj_frame = np.vstack([ self.hand_pos_obj_frame[obj.label] for i in range(self.n_layers) for j in range(self.n_layers) ])
            ray_directions_obj_frame = np.vstack([rot @ ray_dir for ray_dir in self.ray_directions_list])    
            try:
                self.impact_locations[obj.label], _, _ = mesh.ray.intersects_location(ray_origins=ray_origins_obj_frame,
                                                                                ray_directions=ray_directions_obj_frame,
                                                                                multiple_hits=False)
            except:
                self.impact_locations[obj.label] = []
            new = True
        else:
            new = False
            self.impact_locations[obj.label] = []
        if len(self.impact_locations[obj.label])>0:
            self.target_detector.new_impacts(obj, self.impact_locations[obj.label], self.hand_pos_obj_frame[obj.label], self.elapsed)
        # obj.distance_to(self)
        return new, self.impact_locations[obj.label]
    
    def check_target_from_trajectory(self, obj_label, obj_inv_transform, obj_mesh):
        self.get_future_trajectory_points()
        vdir = self.state.get_movement_direction()*np.array([-1,1,1])
        svel = self.state.scalar_velocity
        self.impact_locations_list[obj_label] = []
        i = 0
        rot = obj_inv_transform[:3,:3]
        for point in self.future_points:
            weight = 1/(i+1)
            if i>0:
                vdir = (point - prev_point)
                svel = np.linalg.norm(vdir)
                vdir = vdir/svel
                svel = svel/0.01
            ray_origin = (obj_inv_transform@point)[:3]
            ray_direction = rot @ vdir
            try:
                impact_locations, _, _ = obj_mesh.ray.intersects_location(ray_origins=self.ray_origin,
                                                                        ray_directions=self.ray_directions_list[i],
                                                                        multiple_hits=False)
            except:
                impact_locations = []
            self.impact_locations_list[obj_label] += [loc * weight for loc in impact_locations]
            prev_point = point
            i+=1

    def check_target_from_state(self, obj_label, obj_mesh):
        vdir = self.state.get_movement_direction()*np.array([-1,1,1])
        svel = self.state.scalar_velocity
        point = self.state.position_filtered
        try:
            impact_locations, _, _ = obj_mesh.ray.intersects_location(ray_origins=self.ray_origins,
                                                                    ray_directions=self.ray_directions,
                                                                    multiple_hits=False)
        except:
            impact_locations = []
            
            # impact_locations, ray_viz = self.check_target_from_point(obj_mesh, point, vdir, svel)
        self.impact_locations_list[obj_label] = impact_locations
    
    def check_target_from_point(self, obj_mesh, point, vdir, svel):
        ray_origins_obj_frame, ray_directions_obj_frame, ray_viz = self.get_rays_from_point(point, vdir, svel)
        try:
            impact_locations, _, _ = obj_mesh.ray.intersects_location(ray_origins=ray_origins_obj_frame,
                                                                    ray_directions=ray_directions_obj_frame,
                                                                    multiple_hits=False)
        except:
            impact_locations = []
        return impact_locations, ray_viz

    def get_distance_to(self, obj):
        (closest_points,
        distances,
        triangle_id) = obj.mesh.nearest.on_surface(self.hand_pos_obj_frame[obj.label].reshape(1,3))
        self.target_detector.update_distance_to(obj, distances[0], self.elapsed)
    
    def get_mesh_position(self):
        return self.mesh_position.v
    
    def get_trajectory_points(self):
        return self.state.trajectory.get_xyz_data()
    
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

    def get_target_data(self):
        return self.targets_data
        
class GraspingHandState(State):
    def __init__(self,  position=None, normalized_landmarks=None,  timestamp = None, trajectory = None) -> None:
        super().__init__()
        self.position_raw = Position(position)
        self.normalized_landmarks = normalized_landmarks
        # if landmarks is None:
        #     self.landmarks = np.zeros((21,3))
        # else:
        #     self.landmarks = landmarks
        self.new_position = self.position_raw
        self.new_normalized_landmarks = self.normalized_landmarks
        
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
            self.trajectory = GraspingHandTrajectory.from_state(self)
        else:
            self.trajectory = trajectory
        self.set_updated(True)
        self.set_propagated(False)
        
    @classmethod
    def from_hand_detection(cls, hand_detection: HandPrediction, timestamp = 0):
        return cls(hand_detection.position, hand_detection.normalized_landmarks, timestamp)
    
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
        trajectory = GraspingHandTrajectory.from_dataframe(df)
        first_position, first_timestamp = trajectory[0]
        print(f'next hand position : {first_position, first_timestamp}')
        
        return cls(first_position, timestamp=first_timestamp, trajectory=trajectory)
    
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
        self.set_updated(True)
        self.set_propagated(False)
        
    def propagate(self, timestamp):
        elapsed = timestamp - self.last_timestamp        
        self.last_timestamp = timestamp
        self.propagate_position(elapsed)
        if self.normalized_landmarks is not None:
            self.propagate_normalized_landmarks(elapsed)
        self.set_updated(False)
        self.set_propagated(True)
        print(f'propagated hand position : {self.position_raw, self.last_timestamp}')
        
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
        self.compute_future_points()
        self.set_updated(False)
        self.set_propagated(True)
    
    def propagate_only_position(self):
        if self.was_updated():                   
            self.position_raw = self.new_position
            self.trajectory.add(self)
            self.trajectory.fit(10)
            # print('was_updated')
        else:
            # print('was extrapolated')
            extrapolated_position = self.trajectory.extrapolate(self.last_timestamp)
            # print(f'extrapolated_position : {extrapolated_position}')
            self.position_raw = Position(extrapolated_position)
            self.trajectory.add(self, extrapolated=True)
            
    def compute_future_points(self, timespand = 1):
        timestamps = np.arange(self.last_timestamp, self.last_timestamp+timespand, 0.1)
        self.future_points=[]
        for timestamp in timestamps:
            future_point = self.trajectory.extrapolate(timestamp)
            # print(f'future_point : {future_point}')
            self.future_points.append(future_point*np.array([-1,1,1]))
            
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
        return self.trajecory.__next__()

    def __getitem__(self, index):
        return self.trajectory[index]
    