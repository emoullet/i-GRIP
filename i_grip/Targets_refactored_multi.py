import numpy as np  
import time
from pysdf import SDF
from i_grip.utils2 import *
from i_grip import Objects as ob
from i_grip import Hands_refactored as ha

class DataWindow:
    def __init__(self, size:int, label:str) -> None:
        self.size = size # nb iterations or time limit ?
        self.data = list()
        self.nb_samples = 0
        self.label = label
    
    def queue(self, new_data):
        self.data.append(new_data)
        if len(self.data)>= self.size:
            del self.data[0]
        else:
            self.nb_samples+=1
    def get_data(self):
        return self.data
            
class HandConeImpactsWindow(DataWindow):
    def __init__(self, size: int, label:str) -> None:
        super().__init__(size, label)
        self.nb_impacts = 0

    # def mean(self):
    #     if self.nb_samples==0:
    #         return None
    #     sum = 0
    #     for i in range(self.data):
    #         sum+=np.mean(self.data[i])
    #     return sum/self.nb_samples

    def multi_queue(self, new_data):
        for data in new_data:
            self.queue(data)
    
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
    
class RealTimeWindow(DataWindow):
    def __init__(self, size: int, label:str) -> None:
        print('new RealTimeWindow', label)
        super().__init__(size, label)
        self.nb_samples = 0
        self.timestamps = []
        self.poly_coeffs = None
        self.der_poly_coeffs = None
        self.der_data = []
        self.interpolated_data = []
        self.extrapolated_data = []
        self.last_timestamp = 0
        self.proportion_index = 2
        self.min_len = 8 * self.proportion_index
        self.start_index_interpolation = size % self.proportion_index
    def queue(self, new_data:tuple, time_type = 'elapsed'):
        
        self.data.append(new_data[0])
        if time_type == 'elapsed':
            self.timestamps.append(new_data[1])
            self.timestamps = [t-self.timestamps[-1] for t in self.timestamps]
        else:
            dt = new_data[1] - self.last_timestamp
            self.timestamps = [t- dt for t in self.timestamps]
            self.timestamps.append(0)
            self.last_timestamp = new_data[1]
            
        if len(self.data)>= self.size:
            del self.data[0]
            del self.timestamps[0]
        else:
            self.nb_samples+=1
        
    def interpolate(self):
        # compute polynomial fit of data as a function of timestamps
        if len(self.data) < self.min_len:
            self.interpolated_data = self.data
            return 
        else:
            t1 = list(self.timestamps)
            d1 = list(self.data)
            
            t = np.array(self.timestamps[self.start_index_interpolation::self.proportion_index])
            d = np.array(self.data[self.start_index_interpolation::self.proportion_index])
            t_uncut = np.array(self.timestamps)
            #keep one out of proportion_index points
            self.poly_coeffs = np.polynomial.polynomial.polyfit(t, d, 2)
            self.interpolated_data = np.polynomial.polynomial.polyval(t_uncut, self.poly_coeffs)
    
    def differentiate(self):
        # use self.poly_coeffs to compute derivative of data
        if self.poly_coeffs is None:
            # print('No polynomial fit found, please use interpolate() before trying to differentiate()')
            return 
        else:
            self.der_poly_coeffs = np.polynomial.polynomial.polyder(self.poly_coeffs)
        # use self.der_poly_coeffs to compute derivative of data
        self.der_data = np.polynomial.polynomial.polyval(self.timestamps, self.der_poly_coeffs)
    
    def extrapolate(self):
        # compute polynomial fit of data as a function of timestamps
        if len(self.data) < self.min_len:
            self.extrapolated_data = [0 for i in range(len(self.data))]
            self.extrapolated_timestamps = self.timestamps
            return 
        else:
            self.extrapolated_timestamps = np.arange(0, 0.3, 0.01)
            self.extrapolated_data = np.polynomial.polynomial.polyval(self.extrapolated_timestamps, self.poly_coeffs)
    
    def analyse(self, interpolate = True, differentiate = True, extrapolate = True):
        if interpolate:
            self.interpolate()
        if differentiate:
            self.differentiate()
        if extrapolate:
            self.extrapolate()
        
    def get_mean_derivative(self, sub_window_size = 5):
        if self.der_data is None:
            # print('No derivative found, please use differentiate() before trying to get_mean_derivative()')
            return None
        else:
            return np.mean(self.der_data[-sub_window_size:])
    
    def get_zero_time(self):
        # compute time when distance is zero
        # print(f'get_zero_time: {self.poly_coeffs}')
        if self.poly_coeffs is None:
            # print('No polynomial fit found, please use interpolate() before trying to get_zero_time()')
            return 0
        else:
            roots = np.roots(self.poly_coeffs)
            # print(f'roots: {roots}')
            #test if roots are real and positive
            good_roots = [root for root in roots if root.imag == 0 and root.real > 0]
            # print(f'good_roots: {good_roots}')
            if len(good_roots) == 0:
                zero_time = 0
            else:
                zero_time = good_roots[0]
            # print('zero_time', zero_time)
            return zero_time
    
    def get_zero_time_explore(self):
        if self.poly_coeffs is None:
            # print('No polynomial fit found, please use interpolate() before trying to get_zero_time()')
            return 0
        #extrapolate to find time when distance is zero
        extrapolated_timestamps = np.arange(0, 2, 0.01)
        extrapolated_data = np.polynomial.polynomial.polyval(extrapolated_timestamps, self.poly_coeffs)
        # Find index where values switch sign in extrapolated_data
        sign_switch_indices = np.where(np.diff(np.sign(extrapolated_data)))[0]
        if len(sign_switch_indices) >0:
            zero_time = extrapolated_timestamps[sign_switch_indices[0]]
        else:
            zero_time = 0
        return zero_time
    
    def __str__(self) -> str:
        return f'{self.label} - data: {self.data} - timestamps: {self.timestamps} - nb_samples: {self.nb_samples}'
    
class TargetDetector(Timed):
    
    _METRICS_COLORS = ['brown', 'grey', 'black']
    def __init__(self, hand,  window_size = 20, plotter = None, timestamp=None) -> None:
        super().__init__(timestamp=timestamp)
        self.hand = hand
        self.window_size = window_size
        self.potential_targets:dict(Target) = {}
        self.objects = {}
        self.label = hand.label+'_target_detector'
        self.hand_label = hand.label
        self.plotter = plotter
        self.hand_color = hand.plot_color 
        
        self.target_from_distance_window = RealTimeWindow(window_size, self.hand_label+'_target_from_distance')
        self.target_from_impacts_window = RealTimeWindow(window_size, self.hand_label+'_target_from_impacts')
        self.target_from_distance_derivative_window = RealTimeWindow(window_size, self.hand_label+'_target_from_distance_derivative')
        
        
        self.hand_x_window = RealTimeWindow(20, self.hand_label+'_x')
        self.hand_y_window = RealTimeWindow(20, self.hand_label+'_y')
        self.hand_z_window = RealTimeWindow(20, self.hand_label+'_z')
        
        self.hand_scalar_velocity_window = RealTimeWindow(50, self.hand_label+'_hand_scalar_velocity')
        
        self.target_from_distance_confidence_window = RealTimeWindow(window_size, self.hand_label+'_target_from_distance_confidence')
        self.target_from_impacts_confidence_window = RealTimeWindow(window_size, self.hand_label+'_target_from_impacts_confidence')
        self.target_from_distance_derivative_confidence_window = RealTimeWindow(window_size, self.hand_label+'_target_from_distance_derivative_confidence')
        
        self.most_probable_target_window = RealTimeWindow(window_size, self.hand_label+'_most_probable_target')
        self.check_all_targets_time_window = RealTimeWindow(window_size*2, self.hand_label+'_check_all_targets_time')
        
        self.relative_distance_threshold = 0.1
        self.hand_scalar_velocity = 0
        # self.derivative_min_cutoff = derivative_min_cutoff
        # self.derivative_beta = derivative_beta
        # self.derivative_derivate_cutoff = derivative_derivate_cutoff
        #create a list of plotters for each target
        self.define_velocity_cone()
    
            
    def new_target(self, obj:ob.RigidObject):
        print(f'new target {obj.label} for target detector {self.hand_label}')
        self.potential_targets[obj.label] = Target(self.hand, obj, index = len(self.potential_targets))
        self.objects[obj.label] = obj
        
    def poke_target(self, obj:ob.RigidObject):
        if not obj.label in self.potential_targets:
            self.new_target(obj)
    
    def define_velocity_cone(self,cone_max_length = 500, cone_min_length = 50, vmin=30, vmax=200, cone_max_diam = 200,cone_min_diam = 50, n_layers=6):
        self.cone_angle = np.pi/8
        self.total_nb_ray_layers = n_layers
        self.nb_ray_layers_per_point = n_layers
        self.cone_lenghts_spline = spline(vmin,vmax, cone_min_length, cone_max_length)
        # self.cone_diam_spline = spline(vmin,vmax, cone_min_diam, cone_max_diam)
        self.cone_diam_spline = spline(vmin,vmax, cone_max_diam, cone_min_diam)
        
    
    def make_rays_from_trajectory(self):
        future_points = self.hand.get_future_trajectory_points()
        vdir = self.hand.get_movement_direction()*np.array([-1,1,1])
        svel = self.hand.get_scalar_velocity()
        ray_origins_list = []
        ray_directions_list = []
        nb_ray_layers_per_point = max(2,int(self.total_nb_ray_layers/(len(future_points)+1)))
        i=0
        for point in future_points:
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
                svel = (svel*(1+i*0.05))/0.01
            ray_origins, ray_directions = self.get_rays_from_point(point, vdir, svel, nb_ray_layers_per_point)
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
    
    def get_rays_from_point(self, point, vdir, svel, nb_ray_layers):
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

        ray_directions_list = [ vdir*cone_len + (-cone_diam/2+i*cone_diam/(nb_ray_layers-1)) * orthogonal_unit_vector+ (-cone_diam/2+j*cone_diam/(nb_ray_layers-1)) * orthogonal_unit_vector2 for i in range(nb_ray_layers) for j in range(nb_ray_layers) ] 

        ray_origins = np.vstack([point for i in range(nb_ray_layers) for j in range(nb_ray_layers) ])
        ray_directions = np.vstack(ray_directions_list)
        return ray_origins, ray_directions

    def get_rays(self):
        self.make_rays_from_trajectory()
        return self.ray_visualize

    def does_target_need_update(self, obj_label):
        return self.hand.was_mesh_updated() or self.objects[obj_label].was_mesh_updated()
    
    def check_all_targets(self, timestamp = None):
        # self.set_timestamp(self.hand.timestamp)
        # self.set_timestamp(timestamp)
        t= time.time()
        all_impacts = []
        target_labels = self.potential_targets.copy().keys()
        for target_label in target_labels:
            if self.potential_targets[target_label].needs_update():
                all_impacts += self.potential_targets[target_label].update(self.ray_origins, self.ray_directions, self.timestamp)
        # print(f'check all targets {(time.time()-t)*1000} ms')
        self.check_all_targets_time_window.queue(((time.time()-t)*1000, self.get_elapsed()))
        if len(all_impacts)>0:
            return np.vstack(all_impacts)
        else:
            return None
        
    def update_target_position(self, obj_label, obj_position):
        self.potential_targets[obj_label].set_position(obj_position)
            
    def update_distance_to(self, obj:ob.RigidObject, new_distance, elapsed):
        # print(f'update distance from {self.hand_label} to {obj.label} {new_distance} {elapsed}')
        label = obj.label
        if not label in self.potential_targets:
            self.new_target(obj)
        self.potential_targets[label].update_distance(new_distance, elapsed)
        
    def get_most_probable_target(self, timestamp=None):
        self.set_timestamp(timestamp)
        elapsed = self.get_elapsed()
        t= time.time()
        for label,  target in self.potential_targets.items():
            if target.needs_analysis():
            # print(f'analyse target {label}')
                target.analyse()
            # print(f'send plots for target {label}')
            # self.send_plots(label)
        print(f'analyse all targets {(time.time()-t)*1000} ms')
        t= time.time()
        self.hand_x_window.queue((self.hand.mesh_position.x, elapsed))
        self.hand_y_window.queue((self.hand.mesh_position.y, elapsed))
        self.hand_z_window.queue((self.hand.mesh_position.z, elapsed))
        
        self.hand_x_window.analyse()
        self.hand_y_window.analyse()
        self.hand_z_window.analyse()
        
        hand_vx = self.hand_x_window.get_mean_derivative()
        hand_vy = self.hand_y_window.get_mean_derivative()
        hand_vz = self.hand_z_window.get_mean_derivative()
        
        self.hand_scalar_velocity  = np.sqrt(hand_vx**2 + hand_vy**2 + hand_vz**2)
                
        delta_visu = 0.1
        
        target_from_impacts, target_from_impacts_index, target_from_impacts_confidence = self.get_most_probable_target_from_impacts()
        self.target_from_impacts_window.queue( (target_from_impacts_index+2*delta_visu, elapsed) )
        self.target_from_impacts_confidence_window.queue( (target_from_impacts_confidence, elapsed) )
        
        target_from_distance, target_from_distance_index, target_from_distance_confidence = self.get_most_probable_target_from_distance()
        self.target_from_distance_window.queue( (target_from_distance_index+delta_visu, elapsed))
        self.target_from_distance_confidence_window.queue( (target_from_distance_confidence, elapsed) )
        
        target_from_distance_derivative, target_from_distance_derivative_index, target_from_distance_derivative_confidence = self.get_most_probable_target_from_distance_derivative()
        self.target_from_distance_derivative_window.queue( (target_from_distance_derivative_index-delta_visu, elapsed) )
        self.target_from_distance_derivative_confidence_window.queue( (target_from_distance_derivative_confidence, elapsed) )
        
        self.hand_scalar_velocity_window.queue((self.hand_scalar_velocity , elapsed))
        
        self.send_plots()
        
        if self.hand_scalar_velocity > 10:
            if target_from_impacts is not None:
                if target_from_impacts_confidence < target_from_distance_derivative_confidence:
                    most_probable_target = target_from_distance_derivative
                    most_probable_target_confidence = target_from_distance_derivative_confidence
                    most_probable_target_index = target_from_distance_derivative_index
                else:
                    most_probable_target = target_from_impacts
                    most_probable_target_confidence = target_from_impacts_confidence
                    most_probable_target_index = target_from_impacts_index
            else:
                most_probable_target = target_from_distance_derivative
                most_probable_target_confidence = target_from_distance_derivative_confidence
                most_probable_target_index = target_from_distance_derivative_index
        else:
            most_probable_target = target_from_distance
            most_probable_target_confidence = target_from_distance_confidence
            most_probable_target_index = target_from_distance_index
        
        # if target_from_distance.get_distance_to_hand()<self.distance_threshold:
        #     most_probable_target = target_from_distance
        self.most_probable_target_window.queue((most_probable_target_index, elapsed))
        print(f'get most probable target {(time.time()-t)*1000} ms')
        
        return most_probable_target, self.potential_targets

    def compute_minimum_distance_between_targets(self):
        min_distance_between_targets = 999999999999
        for label1, target1 in self.potential_targets.items():
            for label2, target2 in self.potential_targets.items():
                if label1 != label2:
                    dist = Position.distance(target1.position, target2.position)
                    if dist < min_distance_between_targets:
                        min_distance_between_targets = dist
        return min_distance_between_targets
            
    def get_most_probable_target_from_impacts(self):
        if self.potential_targets:
            confidence = 0
            n_impacts = {}
            n_tot = 0
            to_del_keys=[]
            for lab, target in self.potential_targets.items():
                n_impacts[lab] = target.projected_collison_window.get_nb_impacts()
                # if n_impacts[lab]<=0:
                #     to_del_keys.append(lab)
                n_tot +=n_impacts[lab]

            # for key in to_del_keys:
            #     print(f'delete target {key}')
            #     del self.potential_targets[key]
                
            if n_tot == 0:
                most_probable_target =  None                
            else:
                for lab in self.potential_targets:
                    tar_confidence = n_impacts[lab]/n_tot
                    self.potential_targets[lab].set_impact_ratio(tar_confidence)
                    if tar_confidence > confidence:
                        confidence = tar_confidence
                        most_probable_target = self.potential_targets[lab]
        else:
            most_probable_target =  None
        if most_probable_target is not None:
            return most_probable_target, most_probable_target.get_index(), confidence
        else:
            return None, 0, 0
    
    def get_most_probable_target_from_distance_derivative(self):
        confidence = 0
        most_probable_target = None
        if self.potential_targets.values():
            max_distance_derivative = 0
            for target in self.potential_targets.values():
                dist_der = target.get_mean_distance_derivative()   
                if dist_der is not None:
                    if max_distance_derivative == 0 or dist_der > max_distance_derivative:
                        max_distance_derivative = dist_der
                        most_probable_target = target
            if self.hand_scalar_velocity != 0 and max_distance_derivative > 0 :
                confidence = max_distance_derivative/self.hand_scalar_velocity
        if most_probable_target is not None:
            return most_probable_target, most_probable_target.get_index(), confidence
        else:
            return None, 0, 0

    def get_most_probable_target_from_distance(self):
        confidence = 1
        most_probable_target = None
        if self.potential_targets.values():
            min_distance = 999999999999
            for target in self.potential_targets.values():
                dist = target.get_distance_to_hand()
                if min_distance == 999999999999 or dist < min_distance:
                    min_distance = dist
                    most_probable_target = target
            min_distance_between_targets = self.compute_minimum_distance_between_targets()
            if min_distance < self.relative_distance_threshold*min_distance_between_targets:
                confidence = 1
            elif min_distance > min_distance_between_targets:
                confidence = 0
            else:
                confidence = 1 - (min_distance - self.relative_distance_threshold*min_distance_between_targets)/(min_distance_between_targets*(1-self.relative_distance_threshold))
        if most_probable_target is not None:
            return most_probable_target, most_probable_target.get_index(), confidence
        else:
            return None, 0, 0
    
    # def set_hand_absolute_position(self, hand_pos):
    # #     self.hand_pos = hand_pos
        
    # def set_hand_scalar_velocity(self, hand_scalar_velocity):
    #     self.hand_scalar_velocity = hand_scalar_velocity
       #def set_hand_pos_vel(self, hand_pos, hand_vel):
        #self.hand_pos = hand_pos
        #self.hand_vel = hand_vel

    #def compute_time_before_impact(self):
    #    for tar in self.potential_targets:
    #         """

    def send_plots(self):
        
        if self.plotter is not None:
            
            t = time.time()
            to_plots = []
            for label,  target in self.potential_targets.items():
                if self.plotter is not None:
                    tgp = time.time()
                    to_plots += target.get_plots()
                    print(f'get plots for target {target.label} {(time.time()-tgp)*1000} ms')
            #         tpp = time.time()
            #         for to_plot in to_plots:
            #             # to_plot['label'] = label
            #             # to_plot['color'] = self.hand_color
            #             # to_plot['time'] = timestamp
            #             self.plotter.plot(to_plot)
            #         if (time.time()-tpp)*1000>10:
            #             print(to_plots)
            #             # jk
            #         print(f'send plots for target {target.label} {(time.time()-tpp)*1000} ms')
            # print(f'plot all targets {(time.time()-t)*1000} ms')
            
            t = time.time()
            to_plot_target_from_distance = dict(color = TargetDetector._METRICS_COLORS[0],
                            label = self.label,
                            hand_label = self.hand_label, 
                            x = self.target_from_distance_window.timestamps, 
                            y = self.target_from_distance_window.data, 
                            plot_marker = 's', 
                            time = t, 
                            plot_type='',
                            plot_target = 'Targets',
                            metric_label = 'Distance')
            to_plot_target_from_distance_derivative = dict(color = TargetDetector._METRICS_COLORS[1],
                            label = self.label,
                            hand_label = self.hand_label, 
                            x = self.target_from_distance_derivative_window.timestamps, 
                            y = self.target_from_distance_derivative_window.data, 
                            plot_marker = '*', 
                            time = t, 
                            plot_type='',
                            plot_target = 'Targets',
                            metric_label = 'Distance derivative')
            to_plot_target_from_impacts = dict(color = TargetDetector._METRICS_COLORS[2],
                            label = self.label,
                            hand_label = self.hand_label, 
                            x = self.target_from_impacts_window.timestamps, 
                            y = self.target_from_impacts_window.data, 
                            plot_marker = '^', 
                            time = t, 
                            plot_type='',
                            plot_target = 'Targets',
                            metric_label = 'Impacts')
            
            to_plot_most_probable_target = dict(color = self.hand_color,
                            label = self.label,
                            hand_label = self.hand_label, 
                            x = self.most_probable_target_window.timestamps, 
                            y = self.most_probable_target_window.data, 
                            plot_marker = 'o', 
                            time = t, 
                            plot_type='',
                            plot_target = 'Targets',
                            metric_label = 'Most probable target')
            
            
            to_plot_confidence_from_distance = dict(color = TargetDetector._METRICS_COLORS[0],
                            label = self.label,
                            hand_label = self.hand_label, 
                            x = self.target_from_distance_confidence_window.timestamps, 
                            y = self.target_from_distance_confidence_window.data, 
                            plot_marker = 's', 
                            time = t, 
                            plot_type='',
                            plot_target = 'Confidence',
                            metric_label = 'Distance')
            to_plot_confidence_from_distance_derivative = dict(color = TargetDetector._METRICS_COLORS[1],
                            label = self.label,
                            hand_label = self.hand_label, 
                            x = self.target_from_distance_derivative_confidence_window.timestamps, 
                            y = self.target_from_distance_derivative_confidence_window.data, 
                            plot_marker = '*', 
                            time = t, 
                            plot_type='',
                            plot_target = 'Confidence',
                            metric_label = 'Distance derivative')
            to_plot_confidence_from_impacts = dict(color = TargetDetector._METRICS_COLORS[2],
                            label = self.label,
                            hand_label = self.hand_label, 
                            x = self.target_from_impacts_confidence_window.timestamps, 
                            y = self.target_from_impacts_confidence_window.data, 
                            plot_marker = '^', 
                            time = t, 
                            plot_type='',
                            plot_target = 'Confidence',
                            metric_label = 'Impacts')          
            
            to_plot_hand_scalar_velocity = dict(color = self.hand_color,
                            label = self.label,
                            hand_label = self.hand_label, 
                            x = self.hand_scalar_velocity_window.timestamps, 
                            y = self.hand_scalar_velocity_window.data, 
                            plot_marker = '*', 
                            time = t, 
                            plot_type='',
                            plot_target = 'Distance derivative',
                            metric_label = 'Scalar velocity')  
            
            to_plot_check_all_targets_time = dict(color = self.hand_color, 
                            label = self.label,
                            hand_label = self.hand_label,
                            x = self.check_all_targets_time_window.timestamps,
                            y = self.check_all_targets_time_window.data,
                            plot_marker = 'o',
                            time = t,
                            plot_type = '',
                            plot_target = 'Computation Times',
                            metric_label = 'Check all targets')
            print(f'mean_check_all_targets_time {np.mean(self.check_all_targets_time_window.data)}')
            
            to_plots += [to_plot_target_from_distance, to_plot_target_from_distance_derivative, to_plot_target_from_impacts, to_plot_confidence_from_distance, to_plot_confidence_from_distance_derivative, to_plot_confidence_from_impacts, to_plot_hand_scalar_velocity, to_plot_most_probable_target, to_plot_check_all_targets_time]
            
            self.plotter.plot(to_plots)
            # for to_plot in to_plots:
            #     self.plotter.plot(to_plot)
            
    
class Target(Timed):
    
    _TARGETS_COLORS = ['green',  'orange', 'purple', 'pink', 'brown', 'grey', 'black']
    
    def __init__(self,hand:ha.GraspingHand, object:ob.RigidObject,  impacts=None,  analysis_window_size=10, visu_window_size = 40, index = 0) -> None:
        super().__init__()
        self.hand = hand
        self.object = object
        
        self.index = index+1
        self.color = Target._TARGETS_COLORS[index]
        self.obj_label = object.name
        self.hand_label = hand.label
        self.label = self.obj_label + '_from_' + self.hand_label
        self.window_size = visu_window_size
        self.position = self.object.get_position()
        
        self.set_timestamp( max(self.hand.timestamp, self.object.timestamp))
        
        self.projected_collison_window = HandConeImpactsWindow(analysis_window_size, self.label+'_projected impacts')
        self.distance_window = RealTimeWindow(20, self.label+'_distance')
        self.time_to_target_distance_window = RealTimeWindow(visu_window_size, self.label+'_time to target distance')
        self.time_to_target_impacts_window = RealTimeWindow(visu_window_size, self.label+'_time to target impacts')
        self.distance_mean_derivative_window = RealTimeWindow(visu_window_size, self.label+'_mean derivative')
        self.nb_impacts_window = RealTimeWindow(visu_window_size, self.label+'_nb impacts')
        
        self.update_compute_time_window = RealTimeWindow(visu_window_size, self.label + '_update_time')
        self.check_impacts_compute_time_window = RealTimeWindow(visu_window_size, self.label+'_check_impacts')
        self.anaysis_compute_time_window = RealTimeWindow(visu_window_size, self.label+ '_analysis')
        
        self.ratio=0
        self.distance_mean_derivative = 0
        self.time_before_impact_distance = 0
        self.time_before_impact_zone = 0
        self.distance_to_hand = 0
        self.grip = 'None'
        if impacts is not None :
            self.projected_collison_window.queue(impacts)
            self.predicted_impact_zone = Position(self.projected_collison_window.mean())
            self.find_grip()
        else: 
            self.predicted_impact_zone = None
        self.signed_distance_finder = SDF(self.object.mesh.vertices, self.object.mesh.faces)
        self.update_distance()
        self.updated = True
        self.analysed = False
    
    def update_distance(self):
        inv_trans = self.object.inv_mesh_transform
        hand_pos_obj_frame = (inv_trans@self.hand.mesh_position.ve)[:3]
        self.relative_hand_pos = Position(hand_pos_obj_frame)
        
        # update distance to target
        t = time.time()
        # (_, distances, _) = self.object.mesh.nearest.on_surface(hand_pos_obj_frame.reshape(1,3))
        # new_distance = distances[0]
        new_distance = -self.signed_distance_finder(hand_pos_obj_frame.reshape(1,3))[0]
        # print('new_distance', new_distance)
        self.distance_to_hand = new_distance
        # new_distance = 100
        print(f'compute time for distance {(time.time()-t)*1000} ms')
        # print('new_distance', new_distance)
        # print('label', self.obj_label, 'new_distance', new_distance, 'elapsed', elapsed)
        self.distance_window.queue((new_distance, self.get_elapsed()))
    
    def was_updated(self):
        return self.updated
    
    def needs_update(self):
        my_bool = self.hand.timestamp > self.timestamp or self.object.timestamp > self.timestamp
        print(f'{self.hand_label} - {self.obj_label} target needs update: {my_bool}')
        return my_bool
    
    def update(self, ray_origins, ray_directions, timestamp):
        self.set_timestamp(max(self.hand.timestamp, self.object.timestamp))
        t = time.time()
        impacts = self.check_impacts(ray_origins, ray_directions)
        self.check_impacts_compute_time_window.queue(((time.time()-t)*1000, self.get_elapsed()))
        t = time.time()
        self.update_distance()
        self.update_compute_time_window.queue(((time.time() - t)*1000, self.get_elapsed()))
        self.updated = True
        self.analysed = False
        return impacts
    
    def check_impacts(self, ray_origins, ray_directions):
        t = time.time()
        # print(f'check impacts for {self.object.label}')
        inv_trans = self.object.inv_mesh_transform
        # update impact locations
        ray_origins_obj_frame = []
        impacts = []
        ray_origins_obj_frame = np.vstack([(inv_trans@np.append(ray_origin, 1))[:3] for ray_origin in ray_origins])
        
        rot = inv_trans[:3,:3]
        ray_directions_obj_frame = np.vstack([rot @ ray_dir for ray_dir in ray_directions])    
        try:
            impacts, _, _ = self.object.mesh.ray.intersects_location(ray_origins=ray_origins_obj_frame,
                                                                        ray_directions=ray_directions_obj_frame,
                                                                        multiple_hits=False)
        except:
            impacts = []
        self.projected_collison_window.queue(impacts)
        self.predicted_impact_zone = Position(self.projected_collison_window.mean())
        self.nb_impacts_window.queue((self.projected_collison_window.nb_impacts, self.get_elapsed()))
        # print(f'window {self.nb_impacts_window}')
        if len(impacts)>0:
            impacts_scene_frame = [(self.object.get_mesh_transform()@np.append(impact, 1))[:3] for impact in impacts]
        else:
            impacts_scene_frame = []
        print(f'compute time hand {self.hand_label} target {self.obj_label} for impacts {(time.time()-t)*1000} ms')
        # self.check_impacts_compute_time_window.queue((time.time()-t, self.get_elapsed()))
        return impacts_scene_frame
    
    def check_impacts_old(self, ray_origins, ray_directions):
        t = time.time()
        if not (self.hand.was_mesh_updated() or self.object.was_mesh_updated()):
            print(f'no need to check target for {self.object.label}')
            new = False
            impacts = []
        else:
            # print(f'check impacts for {self.object.label}')
            inv_trans = self.object.inv_mesh_transform
            # update impact locations
            ray_origins_obj_frame = []
            impacts = []
            ray_origins_obj_frame = np.vstack([(inv_trans@np.append(ray_origin, 1))[:3] for ray_origin in ray_origins])
            
            rot = inv_trans[:3,:3]
            ray_directions_obj_frame = np.vstack([rot @ ray_dir for ray_dir in ray_directions])    
            try:
                impacts, _, _ = self.object.mesh.ray.intersects_location(ray_origins=ray_origins_obj_frame,
                                                                            ray_directions=ray_directions_obj_frame,
                                                                            multiple_hits=False)
            except:
                impacts = []
        self.projected_collison_window.queue(impacts)
        self.predicted_impact_zone = Position(self.projected_collison_window.mean())
        self.nb_impacts_window.queue((self.projected_collison_window.nb_impacts, self.get_elapsed()))
        # print(f'window {self.nb_impacts_window}')
        if len(impacts)>0:
            impacts_scene_frame = [(self.object.get_mesh_transform()@np.append(impact, 1))[:3] for impact in impacts]
        else:
            impacts_scene_frame = []
        print(f'compute time hand {self.hand_label} target {self.obj_label} for impacts {(time.time()-t)*1000} ms')
        return impacts_scene_frame

    def set_impact_ratio(self, ratio):
        self.ratio = ratio

    def set_position(self, position):
        self.position = position
    
    def compute_time_before_impact_zone(self):
        if self.relative_hand_pos is None or self.predicted_impact_zone is None:
            return
        new_distance_to_hand = Position.distance(self.relative_hand_pos, self.predicted_impact_zone)
        if self.elapsed != 0 :
            if self.distance_to_hand is not None:
                distance_der = (self.distance_to_hand - new_distance_to_hand)/self.elapsed
                if distance_der !=0:
                    #self.time_before_impact = new_distance_to_hand
                    self.time_before_impact_zone = new_distance_to_hand/distance_der
                    self.time_to_target_impacts_window.queue((self.time_before_impact_zone, self.elapsed), time_type='elapsed')
                self.distance_derivative = distance_der
            self.distance_to_hand = new_distance_to_hand
            
    def compute_time_before_impact_distance(self):
        self.time_before_impact_distance = self.distance_window.get_zero_time_explore()
        self.time_to_target_distance_window.queue((self.time_before_impact_distance, self.elapsed), time_type='elapsed')
        # print('new_distance_to_hand',new_distance_to_hand)
        # print('time to impact', int(self.time_before_impact ), 'ms')
        
    def needs_analysis(self):
        return not self.analysed
    
    def analyse(self):
        # print('analyse target', self.object.label, self.hand_label)
        t=time.time()
        self.distance_window.analyse()
        # self.time_to_target_distance_window.analyse()
        # self.time_to_target_impacts_window.analyse()
        self.compute_time_before_impact_distance()
        # self.compute_time_before_impact_zone()
        self.distance_mean_derivative = -self.distance_window.get_mean_derivative()
        self.distance_mean_derivative_window.queue((self.distance_mean_derivative, self.elapsed), time_type='elapsed')
        if self.predicted_impact_zone is not None:
            self.find_grip()
        self.anaysis_compute_time_window.queue(((time.time()-t)*1000, self.get_elapsed()))
        print(f'analysis comput time for {self.hand_label} - {self.obj_label} : {(time.time()- t )*1000} ms')
        self.analysed = True
    
    def get_distance_to_hand(self):
        return self.distance_to_hand
    
    def get_mean_distance_derivative(self):
        return self.distance_mean_derivative

    def __str__(self) -> str:
        out = 'Target: '+self.object.label + ' - nb impacts: ' + str(self.projected_collison_window.nb_impacts) + ' - ratio: ' + str(self.ratio)
        return out
    
    def get_proba(self):
        return self.ratio

    def get_time_of_impact(self, unit = 'ms'):
        if unit == 'ms':
            return int(self.time_before_impact_distance*1000)
        if unit == 's':
            return int(self.time_before_impact_distance)
    
    def find_grip(self):
        if abs(self.predicted_impact_zone.v[0])>20:
            self.grip = 'PINCH'
        else:
            self.grip = 'PALMAR'

    def get_grip(self):
        return self.grip
    
    def get_info(self):
        return self.object.name, self.grip, self.time_before_impact_distance, self.ratio

    def get_plots(self):      
        
        t = time.time()
        to_plots = []
        to_plot_distance = dict(color = self.color,
                        label = self.label,
                        hand_label = self.hand_label, 
                        object_label = self.obj_label,
                        object_index = self.index,
                        x = self.distance_window.timestamps, 
                        y = self.distance_window.data, 
                        plot_marker = 'o', 
                        time = t, 
                        plot_type='',
                        plot_target = 'Distance')
        to_plots.append(to_plot_distance)
        
        to_plot_distance_interpolated = dict(color = self.color,
                        label = self.label,
                        hand_label = self.hand_label,
                        object_label = self.obj_label,
                        object_index = self.index,
                        x = self.distance_window.timestamps,
                        y = self.distance_window.interpolated_data,
                        plot_marker = '-',
                        time = t,
                        plot_type='_interpolated',
                        plot_target = 'Distance')
        to_plots.append(to_plot_distance_interpolated)
        
        to_plot_distance_extrapolated = dict(color = self.color,
                        label = self.label,
                        hand_label = self.hand_label,
                        object_label = self.obj_label,
                        object_index = self.index,
                        x = self.distance_window.extrapolated_timestamps,
                        y = self.distance_window.extrapolated_data,
                        plot_marker = '--',
                        time = t, 
                        plot_type='_extrapolated',
                        plot_target = 'Distance')
        to_plots.append(to_plot_distance_extrapolated)
        
        to_plot_time_to_target_distance = dict(color = self.color,
                        label = self.label,
                        hand_label = self.hand_label, 
                        object_label = self.obj_label,
                        object_index = self.index,
                        x = self.time_to_target_distance_window.timestamps, 
                        y = self.time_to_target_distance_window.data, 
                        plot_marker = 'o', 
                        time = t, 
                        plot_type='',
                        plot_target = 'Time to impact')
        to_plots.append(to_plot_time_to_target_distance)
        # print(f'shape {x.shape} {y.shape}')
        to_plot_mean_derivative = dict(color = self.color,
                        label = self.label,
                        hand_label = self.hand_label, 
                        object_label = self.obj_label,
                        object_index = self.index,
                        x = self.distance_mean_derivative_window.timestamps, 
                        y = self.distance_mean_derivative_window.data, 
                        plot_marker = 'o', 
                        time = t, 
                        plot_type='',
                        plot_target = 'Distance derivative')
        to_plots.append(to_plot_mean_derivative)
        
        to_plot_nb_impacts = dict(color = self.color,
                        label = self.label,
                        hand_label = self.hand_label, 
                        object_label = self.obj_label,
                        object_index = self.index,
                        x = self.nb_impacts_window.timestamps, 
                        y = self.nb_impacts_window.data, 
                        plot_marker = 'o', 
                        time = t, 
                        plot_type='',
                        plot_target = 'Impacts')
        to_plots.append(to_plot_nb_impacts)
        
        # to_plot_check_compute_time = dict(color = self.color,
        #                 label = self.label,
        #                 hand_label = self.hand_label, 
        #                 object_label = self.obj_label,
        #                 object_index = self.index,
        #                 x = self.check_impacts_compute_time_window.timestamps, 
        #                 y = self.check_impacts_compute_time_window.data, 
        #                 plot_marker = 'o', 
        #                 time = t, 
        #                 plot_type='',
        #                 plot_target = 'Computation Times')
        # to_plots.append(to_plot_check_compute_time)
        
        # to_plot_update_time_compute_time = dict(color = self.color,
        #                 label = self.label,
        #                 hand_label = self.hand_label, 
        #                 object_label = self.obj_label,
        #                 object_index = self.index,
        #                 x = self.update_compute_time_window.timestamps, 
        #                 y = self.update_compute_time_window.data, 
        #                 plot_marker = '*', 
        #                 time = t, 
        #                 plot_type='',
        #                 plot_target = 'Computation Times')
        # to_plots.append(to_plot_update_time_compute_time)
        
        # to_plot_analysis_time_compute_time = dict(color = self.color,
        #                 label = self.label,
        #                 hand_label = self.hand_label, 
        #                 object_label = self.obj_label,
        #                 object_index = self.index,
        #                 x = self.anaysis_compute_time_window.timestamps, 
        #                 y = self.anaysis_compute_time_window.data, 
        #                 plot_marker = '^', 
        #                 time = t, 
        #                 plot_type='',
        #                 plot_target = 'Computation Times')
        # to_plots.append(to_plot_analysis_time_compute_time)
        return to_plots
    
    def get_index(self):
        return self.index
    
class GripSelector:
    def __init__(self) -> None:
        pass