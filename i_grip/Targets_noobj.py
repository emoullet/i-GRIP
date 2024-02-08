import numpy as np  
from i_grip.utils2 import *
from i_grip import Objects as ob

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
        if len(self.data) < 8:
            self.interpolated_data = self.data
            return 
        else:
            t = np.array(self.timestamps)
            d = np.array(self.data)
            self.poly_coeffs = np.polynomial.polynomial.polyfit(t, d, 2)
            self.interpolated_data = np.polynomial.polynomial.polyval(t, self.poly_coeffs)
    
    def differentiate(self):
        # use self.poly_coeffs to compute derivative of data
        if self.poly_coeffs is None:
            # print('No polynomial fit found, please use interpolate() before trying to differentiate()')
            return None
        else:
            self.der_poly_coeffs = np.polynomial.polynomial.polyder(self.poly_coeffs)
        # use self.der_poly_coeffs to compute derivative of data
        self.der_data = np.polynomial.polynomial.polyval(self.timestamps, self.der_poly_coeffs)
    
    def extrapolate(self):
        # compute polynomial fit of data as a function of timestamps
        if len(self.data) < 8:
            self.extrapolated_data = [0 for i in range(len(self.data))]
            self.extrapolated_timestamps = self.timestamps
            return 
        else:
            self.extrapolated_timestamps = np.arange(0, 0.3, 0.01)
            self.extrapolated_data = np.polynomial.polynomial.polyval(self.extrapolated_timestamps, self.poly_coeffs)
    
    def analyse(self):
        self.interpolate()
        self.extrapolate()
        self.differentiate()
        pass
        
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
    
class TargetDetector:
    
    _METRICS_COLORS = ['brown', 'grey', 'black']
    def __init__(self, hand_label, hand_color, window_size = 20, plotter = None) -> None:
        self.window_size = window_size
        self.potential_targets:dict(Target) = {}
        self.label = hand_label+'_target_detector'
        self.hand_label = hand_label
        self.plotter = plotter
        self.hand_color = hand_color 
        
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
        
        self.relative_distance_threshold = 0.1
        self.hand_scalar_velocity = 0
        # self.derivative_min_cutoff = derivative_min_cutoff
        # self.derivative_beta = derivative_beta
        # self.derivative_derivate_cutoff = derivative_derivate_cutoff
        #create a list of plotters for each target
    
    def new_target(self, obj:ob.RigidObject):
        self.potential_targets[obj.label] = Target(obj, hand_label=self.hand_label, index = len(self.potential_targets))
        
    def poke_target(self, obj:ob.RigidObject):
        if not obj.label in self.potential_targets:
            self.new_target(obj)
    
    def new_impacts(self, label, impacts, relative_hand_pos, elapsed, weight = 1):
        relative_hand_pos = Position(relative_hand_pos)
        #print('impacts', impacts)
        if label in self.potential_targets:
            self.potential_targets[label].new_impacts(impacts, relative_hand_pos, elapsed) 
        # elif len(impacts)>0:
        #     self.potential_targets[label] = Target(obj, impacts, relative_hand_pos, hand_label=self.hand_label)   
        
    def update_target_position(self, obj_label, obj_position):
        self.potential_targets[obj_label].set_position(obj_position)
            
    def update_distance_to(self, obj:ob.RigidObject, new_distance, elapsed):
        # print(f'update distance from {self.hand_label} to {obj.label} {new_distance} {elapsed}')
        label = obj.label
        if not label in self.potential_targets:
            self.new_target(obj)
        self.potential_targets[label].update_distance(new_distance, elapsed)
        
    def get_most_probable_target(self, timestamp):
        for label,  target in self.potential_targets.items():
            # print(f'analyse target {label}')
            target.analyse()
            # print(f'send plots for target {label}')
            # self.send_plots(label)
            if self.plotter is not None:
                to_plots = target.get_plots()
                for to_plot in to_plots:
                    # to_plot['label'] = label
                    # to_plot['color'] = self.hand_color
                    # to_plot['time'] = timestamp
                    self.plotter.plot(to_plot)
        self.hand_x_window.queue((self.hand_pos.x, timestamp))
        self.hand_y_window.queue((self.hand_pos.y, timestamp))
        self.hand_z_window.queue((self.hand_pos.z, timestamp))
        
        self.hand_x_window.analyse()
        self.hand_y_window.analyse()
        self.hand_z_window.analyse()
        
        hand_vx = self.hand_x_window.get_mean_derivative()
        hand_vy = self.hand_y_window.get_mean_derivative()
        hand_vz = self.hand_z_window.get_mean_derivative()
        
        self.hand_scalar_velocity  = np.sqrt(hand_vx**2 + hand_vy**2 + hand_vz**2)
                
        delta_visu = 0.1
        
        target_from_impacts, target_from_impacts_index, target_from_impacts_confidence = self.get_most_probable_target_from_impacts()
        self.target_from_impacts_window.queue( (target_from_impacts_index+2*delta_visu, timestamp) )
        self.target_from_impacts_confidence_window.queue( (target_from_impacts_confidence, timestamp) )
        
        target_from_distance, target_from_distance_index, target_from_distance_confidence = self.get_most_probable_target_from_distance()
        self.target_from_distance_window.queue( (target_from_distance_index+delta_visu, timestamp))
        self.target_from_distance_confidence_window.queue( (target_from_distance_confidence, timestamp) )
        
        target_from_distance_derivative, target_from_distance_derivative_index, target_from_distance_derivative_confidence = self.get_most_probable_target_from_distance_derivative()
        self.target_from_distance_derivative_window.queue( (target_from_distance_derivative_index-delta_visu, timestamp) )
        self.target_from_distance_derivative_confidence_window.queue( (target_from_distance_derivative_confidence, timestamp) )
        
        self.hand_scalar_velocity_window.queue((self.hand_scalar_velocity , timestamp))
        
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
        self.most_probable_target_window.queue((most_probable_target_index, timestamp))
        
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
            min_distance_derivative = 0
            for target in self.potential_targets.values():
                dist_der = target.get_mean_distance_derivative()   
                if dist_der is not None:
                    if min_distance_derivative == 0 or dist_der < min_distance_derivative:
                        min_distance_derivative = dist_der
                        most_probable_target = target
            if self.hand_scalar_velocity != 0 and min_distance_derivative > 0 :
                confidence = min_distance_derivative/self.hand_scalar_velocity
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
    def set_hand_absolute_position(self, hand_pos):
        self.hand_pos = hand_pos
    def set_hand_scalar_velocity(self, hand_scalar_velocity):
        self.hand_scalar_velocity = hand_scalar_velocity
       #def set_hand_pos_vel(self, hand_pos, hand_vel):
        #self.hand_pos = hand_pos
        #self.hand_vel = hand_vel

    #def compute_time_before_impact(self):
    #    for tar in self.potential_targets:
    #         """

    def send_plots(self):
        
        if self.plotter is not None:
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
            
            to_plots = [to_plot_target_from_distance, to_plot_target_from_distance_derivative, to_plot_target_from_impacts, to_plot_confidence_from_distance, to_plot_confidence_from_distance_derivative, to_plot_confidence_from_impacts, to_plot_hand_scalar_velocity, to_plot_most_probable_target]
            for to_plot in to_plots:
                self.plotter.plot(to_plot)
            
    
class Target:
    
    _TARGETS_COLORS = ['green',  'orange', 'purple', 'pink', 'brown', 'grey', 'black']
    
    def __init__(self, obj:ob.RigidObject, hand_label = None, impacts=None,  relative_hand_pos=None, analysis_window_size=10, visu_window_size = 40, index = 0) -> None:
        self.index = index+1
        self.color = Target._TARGETS_COLORS[index]
        self.object = obj
        # self.obj_label = obj.name
        self.hand_label = hand_label
        # self.label = self.obj_label + '_from_' + self.hand_label
        self.label = obj + '_from_' + self.hand_label
        self.window_size = visu_window_size
        # self.position = self.object.get_position()
        
        self.projected_collison_window = HandConeImpactsWindow(analysis_window_size, self.label+'_projected impacts')
        self.distance_window = RealTimeWindow(20, self.label+'_distance')
        self.time_to_target_distance_window = RealTimeWindow(visu_window_size, self.label+'_time to target distance')
        self.time_to_target_impacts_window = RealTimeWindow(visu_window_size, self.label+'_time to target impacts')
        self.distance_mean_derivative_window = RealTimeWindow(visu_window_size, self.label+'_mean derivative')
        self.nb_impacts_window = RealTimeWindow(visu_window_size, self.label+'_nb impacts')
        
        self.ratio=0
        self.distance_mean_derivative = 0
        self.time_before_impact_distance = 0
        self.time_before_impact_zone = 0
        self.distance_to_hand = 0
        self.grip = 'None'
        if not (impacts is None or relative_hand_pos is None):
            self.projected_collison_window.queue(impacts)
            self.predicted_impact_zone = Position(self.projected_collison_window.mean())
            self.distance_to_hand = Position.distance(relative_hand_pos, self.predicted_impact_zone)
            self.relative_hand_pos = relative_hand_pos
            self.find_grip()
        else: 
            self.predicted_impact_zone = None
            self.distance_to_hand = None
            self.relative_hand_pos = None
        self.elapsed = 0
    
    def update_distance(self, new_distance, elapsed):
        self.distance_to_hand = new_distance
        # print('label', self.obj_label, 'new_distance', new_distance, 'elapsed', elapsed)
        self.distance_window.queue((new_distance, elapsed), time_type='elapsed')

    def new_impacts(self, impacts, new_relative_hand_pos, elapsed):        
        if elapsed != 0:
            self.elapsed = elapsed
            # self.relative_hand_vel = (new_relative_hand_pos.v - self.relative_hand_pos.v)/elapsed
            self.relative_hand_pos = new_relative_hand_pos
        self.projected_collison_window.queue(impacts)
        self.predicted_impact_zone = Position(self.projected_collison_window.mean())
        self.nb_impacts_window.queue((self.projected_collison_window.nb_impacts, elapsed), time_type='elapsed')
    
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
        #print('new_distance_to_hand',new_distance_to_hand)
        # print('time to impact', int(self.time_before_impact ), 'ms')
    
    def analyse(self):
        self.distance_window.analyse()
        self.time_to_target_distance_window.analyse()
        self.time_to_target_impacts_window.analyse()
        self.compute_time_before_impact_distance()
        self.compute_time_before_impact_zone()
        self.distance_mean_derivative = self.distance_window.get_mean_derivative()
        self.distance_mean_derivative_window.queue((self.distance_mean_derivative, self.elapsed), time_type='elapsed')
        if self.predicted_impact_zone is not None:
            self.find_grip()
    
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
            return int(self.time_before_impact_zone*1000)
        if unit == 's':
            return int(self.time_before_impact_zone)
    
    def find_grip(self):
        if abs(self.predicted_impact_zone.v[0])>20:
            self.grip = 'PINCH'
        else:
            self.grip = 'PALMAR'

    def get_grip(self):
        return self.grip
    
    def get_info(self):
        return self.object.name, self.grip, self.time_before_impact_zone, self.ratio

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
        return to_plots
    
    def get_index(self):
        return self.index
    
class GripSelector:
    def __init__(self) -> None:
        pass