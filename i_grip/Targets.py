import numpy as np  
from i_grip.utils2 import *
from i_grip import Objects as ob
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.nb_samples = 0
        self.timestamps = []
        self.poly_coeffs = None
        self.der_poly_coeffs = None
        self.der_data = None
        self.interpolated_data = []
        self.distance_derivative = 0
        self.extrapolated_data = []
    
    def queue(self, new_data:tuple):
        self.data.append(new_data[0])
        self.timestamps.append(new_data[1])
        self.timestamps = [t-self.timestamps[-1] for t in self.timestamps]
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
            print('No polynomial fit found, please use interpolate() before trying to differentiate()')
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
        
    def get_mean_derivative(self):
        self.interpolate()
        self.differentiate()
        if self.der_data is None:
            print('No derivative found, please use differentiate() before trying to get_mean_derivative()')
            return None
        else:
            return np.mean(self.der_data)
    
    def get_zero_time(self):
        # compute time when distance is zero
        print(f'get_zero_time: {self.poly_coeffs}')
        self.interpolate()
        self.extrapolate()
        if self.poly_coeffs is None:
            print('No polynomial fit found, please use interpolate() before trying to get_zero_time()')
            return None
        else:
            roots = np.roots(self.poly_coeffs)
            print(f'roots: {roots}')
            #test if roots are real and positive
            good_roots = [root for root in roots if root.imag == 0 and root.real > 0]
            print(f'good_roots: {good_roots}')
            if len(good_roots) == 0:
                zero_time = 0
            else:
                zero_time = good_roots[0]
            print('zero_time', zero_time)
            return zero_time
    
    
class TargetDetector:
    def __init__(self, hand_label, hand_color, window_size = 100, plotter = None) -> None:
        self.window_size = window_size
        self.potential_targets:dict(Target) = {}
        self.hand_label = hand_label
        self.plotter = plotter
        self.hand_color = hand_color 
        self.target_from_distance_window = RealTimeWindow(window_size)
        self.target_from_impacts_window = RealTimeWindow(window_size)
        self.target_from_distance_derivative_window = RealTimeWindow(window_size)
        # self.derivative_min_cutoff = derivative_min_cutoff
        # self.derivative_beta = derivative_beta
        # self.derivative_derivate_cutoff = derivative_derivate_cutoff
        #create a list of plotters for each target
    
    def new_target(self, obj:ob.RigidObject):
        self.potential_targets[obj.label] = Target(obj)
        
    def poke_target(self, obj:ob.RigidObject):
        if not obj.label in self.potential_targets:
            self.new_target(obj)
    
    def new_impacts(self, obj:ob.RigidObject, impacts, relative_hand_pos, elapsed, weight = 1):
        label = obj.label
        relative_hand_pos = Position(relative_hand_pos)
        #print('impacts', impacts)
        if label in self.potential_targets:
            self.potential_targets[label].new_impacts(impacts, relative_hand_pos, elapsed) 
        elif len(impacts)>0:
            self.potential_targets[label] = Target(obj, impacts, relative_hand_pos)
            
    def update_distance_to(self, obj:ob.RigidObject, new_distance, elapsed):
        # print(f'update distance from {self.hand_label} to {obj.label} {new_distance} {elapsed}')
        label = obj.label
        if not label in self.potential_targets:
            self.new_target(obj)
        self.potential_targets[label].update_distance(new_distance, elapsed)
        
        
    
    def get_most_probable_target(self):
        for label,  target in self.potential_targets.items():
            target.analyse()
            self.send_plots(label)
        target_from_impacts, target_from_impacts_index = self.get_most_probable_target_from_impacts()
        # target_from_distance, target_from_distance_index = self.get_most_probable_target_from_distance()
        # target_from_distance_derivative, target_from_distance_derivative_index = self.get_most_probable_target_from_distance_derivative()
        most_probable_target = target_from_impacts
        
        return most_probable_target, self.potential_targets
    
    def send_plots(self, label):
        
        if self.plotter is not None:
            t = time.time()
            to_plot = dict(color = self.hand_color,
                           label = self.hand_label, 
                           x = self.potential_targets[label].distance_window.timestamps, 
                           y = self.potential_targets[label].distance_window.data, 
                           plot_marker = 'o', 
                           time = t, 
                           plot_type='',
                           plot_target = 'Distance')
            self.plotter.plot(to_plot)
            
            to_plot = dict(color = self.hand_color,
                           label = self.hand_label,
                           x = self.potential_targets[label].distance_window.timestamps,
                           y = self.potential_targets[label].distance_window.interpolated_data,
                           plot_marker = '-',
                           time = t,
                           plot_type='_interpolated',
                           plot_target = 'Distance')
            self.plotter.plot(to_plot)
            
            to_plot = dict(color = self.hand_color,
                           label = self.hand_label,
                           x = self.potential_targets[label].distance_window.extrapolated_timestamps,
                           y = self.potential_targets[label].distance_window.extrapolated_data,
                           plot_marker = '--',
                           time = t, 
                           plot_type='_extrapolated',
                           plot_target = 'Distance')
            self.plotter.plot(to_plot)
            
            to_plot = dict(color = self.hand_color,
                           label = self.hand_label, 
                           x = self.potential_targets[label].time_to_target_distance_window.timestamps, 
                           y = self.potential_targets[label].time_to_target_distance_window.data, 
                           plot_marker = 'o', 
                           time = t, 
                           plot_type='',
                           plot_target = 'Time to impact')
            self.plotter.plot(to_plot)
    
    def get_most_probable_target_from_impacts(self):
        target_index = 0   
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
                for i, lab in enumerate(n_impacts):
                    if lab == max_ratio_label:
                        target_index = i
                most_probable_target = self.potential_targets[max_ratio_label]
        else:
            most_probable_target =  None
        
        return most_probable_target, target_index   
    
    def get_most_probable_target_from_distance_derivative(self):
        target_index = 0
        most_probable_target = None
        if self.potential_targets:
            min_distance_derivative = 0
            for i, target in enumerate(self.potential_targets):
                dist_der = target.get_mean_distance_derivative()   
                if dist_der is not None:
                    if min_distance_derivative == 0 or dist_der < min_distance_derivative:
                        min_distance_derivative = dist_der
                        most_probable_target = target
                        target_index = i
        return most_probable_target, target_index

    def get_most_probable_target_from_distance(self):
        target_index =0
        most_probable_target = None
        if self.potential_targets:
            min_distance = 0
            for i, target in enumerate(self.potential_targets):
                dist = target.distance_window.data[-1]
                if min_distance == 0 or dist < min_distance:
                    min_distance = dist
                    most_probable_target = target
                    target_index = i
        return most_probable_target, target_index
                

       #def set_hand_pos_vel(self, hand_pos, hand_vel):
        #self.hand_pos = hand_pos
        #self.hand_vel = hand_vel

    #def compute_time_before_impact(self):
    #    for tar in self.potential_targets:
    #         """

class Target:
    def __init__(self, obj:ob.RigidObject, impacts=None,  relative_hand_pos=None,  window_size = 10) -> None:
        self.object = obj
        self.label = obj.label
        self.window_size = window_size
        self.projected_collison_window = HandConeImpactsWindow(window_size)
        self.distance_window = RealTimeWindow(window_size)
        self.time_to_target_distance_window = RealTimeWindow(2*window_size)
        self.time_to_target_impacts_window = RealTimeWindow(2*window_size)
        print(f'POTENTIAL TARGET {self.label}')
        self.ratio=0
        self.distance_derivative = 0
        self.grip = 'None'
        self.time_before_impact_distance = 0
        self.time_before_impact_zone = 0
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
        # create a plotter for this target
        
    #     self.plot= plt.figure()
    #     self.ax = self.plot.add_subplot()
    #     self.plot.title = 'Distance to hand'
    #     self.plotter =animation.FuncAnimation(plt.figure(), self.update_plot, interval=30)
    #     self.plot.show()
    
    # def update_plot(self):
    #     self.ax.clear()
    #     self.ax.plot(self.distance_window.timestamps, self.distance_window.data)
    #     self.ax.plot(self.distance_window.timestamps, self.distance_window.interpolated_data)
    # def add_impacts(self, impacts):
    
    def update_distance(self, new_distance, elapsed):
        self.distance_window.queue((new_distance, elapsed))

    def new_impacts(self, impacts, new_relative_hand_pos, elapsed):        
        if elapsed != 0:
            self.elapsed = elapsed
            # self.relative_hand_vel = (new_relative_hand_pos.v - self.relative_hand_pos.v)/elapsed
            self.relative_hand_pos = new_relative_hand_pos
        self.projected_collison_window.queue(impacts)
        self.predicted_impact_zone = Position(self.projected_collison_window.mean())
    
    def set_impact_ratio(self, ratio):
        self.ratio = ratio

    def compute_time_before_impact_zone(self):
        if self.distance_to_hand is None or self.predicted_impact_zone is None:
            return
        new_distance_to_hand = Position.distance(self.relative_hand_pos, self.predicted_impact_zone)
        if self.elapsed != 0 :
            if self.distance_to_hand is not None:
                distance_der = (self.distance_to_hand - new_distance_to_hand)/self.elapsed
                if distance_der !=0:
                    #self.time_before_impact = new_distance_to_hand
                    self.time_before_impact_zone = new_distance_to_hand/distance_der
                    self.time_to_target_impacts_window.queue((self.time_before_impact_zone, self.elapsed))
                self.distance_derivative = distance_der
            self.distance_to_hand = new_distance_to_hand
            
    def compute_time_before_impact_distance(self):
        self.time_before_impact_distance = self.distance_window.get_zero_time()
        self.time_to_target_distance_window.queue((self.time_before_impact_distance, self.elapsed))
        #print('new_distance_to_hand',new_distance_to_hand)
        # print('time to impact', int(self.time_before_impact ), 'ms')
    
    def analyse(self):
        self.compute_time_before_impact_distance()
        self.compute_time_before_impact_zone()
        self.distance_derivative = self.distance_window.get_mean_derivative()
        self.find_grip()
    
    def get_mean_distance_derivative(self):
        return self.distance_derivative

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
    
    def get_info(self):
        return self.object.name, self.grip, self.time_before_impact, self.ratio

class GripSelector:
    def __init__(self) -> None:
        pass