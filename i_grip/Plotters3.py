#!/usr/bin/env python3

import matplotlib
matplotlib.use('tkAgg')
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class ProcessPlotter(object):
    def __init__(self):
        self.plot_is_new = {}
        self.legend_items = {}
        self.lines  = {}
    def terminate(self):
        plt.close('all')


    def call_back(self):
        data_distance = {}
        data_time = {}
        data_targets = {}
        data_distance_derivative = {}
        for hand in ['left', 'right']:
            data_distance[hand] = []
            data_time[hand] = []
            data_targets[hand] = []
            data_distance_derivative[hand] = []
                        
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                if command['plot_target'] == 'Distance':
                    data_distance[command['hand_label']].append(command)
                elif command['plot_target'] == 'Time to impact':
                    data_time[command['hand_label']].append(command)
                elif command['plot_target'] == 'Targets':
                    data_targets[command['hand_label']].append(command)
                elif command['plot_target'] == 'Distance derivative':
                    data_distance_derivative[command['hand_label']].append(command)
                else:
                    print('plot_target not found')
        for hand in ['left', 'right']:
            if len(data_distance)>0:
                self.plot_metric(data_distance[hand], 'Distance', hand)
            if len(data_time)>0:
                self.plot_metric(data_time[hand], 'Time to impact', hand)
            if len(data_targets)>0:
                self.plot_metric(data_targets[hand], 'Targets', hand)
            if len(data_distance_derivative)>0:
                self.plot_metric(data_distance_derivative[hand], 'Distance derivative', hand)
            
        if len(data_distance)>0 or len(data_time)>0 or len(data_targets)>0:
            self.fig.canvas.draw()
        # if len(data_distance)>0 or len(data_time)>0:
        return True

    def plot_metric(self,commands, metric, hand):
        
        if hand not in self.hands:
            # print('plot_target not found')
            return False
        else:
            i = self.hands.index(hand)
            
        if metric not in self.metrics:
            # print('plot_target not found')
            return False
        else:
            j = self.metrics.index(metric)
            
        metric_hand = metric + hand
        
        all_data = {}
        all_timestamps = {}
        
        for command in commands:
            if command['label'] not in all_timestamps.keys():
                all_data[command['label']] = [command]
                all_timestamps[command['label']] = [command['time']]
            else:
                all_data[command['label']].append(command)
                all_timestamps[command['label']].append(command['time'])
              
        # find all indexes of maximum timestamps
        indexes = {}
        for key in all_timestamps.keys():
            lab_timestamps = all_timestamps[key]
            if len(lab_timestamps)>0:
                max_timestamp = max(lab_timestamps)
                indexes [key]= [i for i, x in enumerate(lab_timestamps) if x == max_timestamp]
            
        for key, data in all_data.items():
            for index in indexes[key]:
                command = data[index]
                x=command['x']
                y=command['y']
                # print(f'command: {command}')
                    
                if 'object_label' in command.keys():
                    object_label = command['object_label']
                    if object_label not in self.legend_items.keys():
                        self.legend_items[object_label] = command['color']
                        self.legend_handles.append(mlines.Line2D([], [], color=command['color'], marker='o', linestyle='None', label=object_label))
                        self.fig.legends=[]
                        self.fig.legend(handles=self.legend_handles, loc='right', bbox_to_anchor=(1., 0.5))
                    # self.axs[i,j].legend()
                    line_label = command['label']+command['plot_type']+metric_hand
                
                elif 'metric_label' in command.keys():
                    print('metric_label found : ', command)
                    metric_label = command['metric_label']
                    if metric_label not in self.legend_items.keys():
                        self.legend_items[metric_label] = command['color']
                        self.legend_handles.append(mlines.Line2D([], [], color=command['color'], marker=command['plot_marker'], linestyle='None', label=metric_label))
                        self.fig.legends=[]
                        self.fig.legend(handles=self.legend_handles, loc='right', bbox_to_anchor=(1., 0.5))
                    line_label = command['metric_label']+metric_hand
                
                # print(f'line_label: {line_label}')
                if line_label not in self.lines.keys():
                    self.lines[line_label] = self.axs[i,j].plot(x, y, command['plot_marker'], color = command['color'], label = line_label)[0]
                else:
                    self.lines[line_label].set_data(x, y)
           
            
    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        
        self.hands= ['left', 'right']
        
        self.metrics = ['Distance', 'Time to impact', 'Distance derivative', 'Targets']
        time_min = -0.8
        time_max = 0.3
        xlims={'Distance':[time_min,time_max+0.3], 'Time to impact':[time_min,time_max], 'Distance derivative':[time_min,time_max], 'Targets':[time_min,time_max]}
        ylims={'Distance':[0,800], 'Time to impact':[-2,2], 'Distance derivative':[-1000,1000], 'Targets':[0,4]}
        
        self.fig, self.axs = plt.subplots(len(self.hands),len(self.metrics))
        self.fig.subplots_adjust(right=0.8)
        self.fig.suptitle('Target detection')
        self.fig.set_size_inches(13, 10.5/2)
        
        for i, hand in enumerate(self.hands):
            self.axs[i,0].set_ylabel(hand)
            for j, metric in enumerate(self.metrics):
                self.axs[i,j].set_xlim(xlims[metric])
                self.axs[i,j].set_ylim(ylims[metric])
        
        for j, metric in enumerate(self.metrics):
            print(f'j: {j}')
            print(f'metric: {metric}')
            self.axs[0,j].set_title(metric)
            self.axs[len(self.hands)-1,j].set_xlabel('Timestamps')
            self.axs[len(self.hands)-1,j].sharex(self.axs[0,j])
            
           
        
        sampled_data_handle=mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='samples')
        interpolated_data_handle=mlines.Line2D([], [], color='black', linestyle='-',  label='interpolated')
        extrapolated_data_handle=mlines.Line2D([], [], color='black', linestyle='--',  label='extrapolated')
        self.legend_handles = [sampled_data_handle, interpolated_data_handle, extrapolated_data_handle]
        self.fig.legend(handles=self.legend_handles, loc='right', bbox_to_anchor=(1., 0.5))
        
        timer = self.fig.canvas.new_timer(interval=30)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()
        
    # def plot_time(self,pipe):
    #     print('starting plotter time ...')
    #     self.pipe2 = pipe
    #     figs_labels = ['Time to impact' ]
    #     self.figs_t = {}
    #     self.axs_t = {}
    #     for label in figs_labels:
    #         self.figs_t[label], self.axs_t[label] = plt.subplots()
    #         timer = self.figs_t[label].canvas.new_timer(interval=20)
    #         self.figs_t[label].suptitle(label)
    #         timer.add_callback(self.call_back_time)
    #     print('...done')
    #     plt.show()
    
    # def call_back_time(self):
    #     while self.pipe2.poll():
    #         command = self.pipe2.recv()
    #         if command is None:
    #             self.terminate()
    #             return False
    #         else:
    #             self.axs_t['Time to impact'].clear()
    #             x = [0,1,2,3,4,5]
    #             y = [0,1,2,np.random(),4,5]
    #             self.axs_t['Time to impact'].plot(x, y)
    #             self.figs_t['Time to impact'].canvas.draw()
    #     return True
    
    def add_subplot(self):
        self.new_ax = self.fig.add_subplot(1,1,1)

class NBPlot(object):
    def __init__(self):
        self.plotter = ProcessPlotter()
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()
        
        # self.plot_pipe_time, plotter_pipe_time = mp.Pipe()
        # self.plot_process_time = mp.Process(
        #     target=self.plotter.plot_time, args=(plotter_pipe_time,), daemon=True)
        # self.plot_process_time.start()

    def plot(self, data, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send(data)
    
    def plot_time(self, data, finished=False):
        send = self.plot_pipe_time.send
        if finished:
            send(None)
        else:
            send(data)