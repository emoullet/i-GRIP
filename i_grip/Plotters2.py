#!/usr/bin/env python3

import matplotlib
matplotlib.use('tkAgg')
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class ProcessPlotter(object):
    def __init__(self):
        self.plot_is_new = {}
        self.targets = {}
        self.lines  = {}
    def terminate(self):
        plt.close('all')


    def call_back(self):
        all_data = {}
        all_timestamps = {}
            
        i = 0
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                if command['label'] not in all_timestamps.keys():
                    all_data[command['label']] = [command]
                    all_timestamps[command['label']] = [command['time']]
                else:
                    all_data[command['label']].append(command)
                    all_timestamps[command['label']].append(command['time'])
                    
            i+=1
        # find all inedexes of maximum timestamps
        indexes = {}
        for key in all_timestamps.keys():
            lab_timestamps = all_timestamps[key]
            if len(lab_timestamps)>0:
                max_timestamp = max(lab_timestamps)
                indexes [key]= [i for i, x in enumerate(lab_timestamps) if x == max_timestamp]
        # print(f'indexes: {indexes}')
        # print(f'all_data: {all_data}')
        if len(indexes)>0:
            self.axs['Distance'].clear()
            
        for key, data in all_data.items():
            for index in indexes[key]:
                command = data[index]
                self.x=command['x']
                self.y=command['y']
                self.axs['Distance'].plot(self.x, self.y, command['plot_marker'], color = command['color'], label = command['label']+command['plot_type'])
                self.axs['Distance'].set_xlabel('Timestamps')
                self.axs['Distance'].set_ylabel('Distance')
                self.axs['Distance'].set_xlim(-0.6,0.6)
                self.axs['Distance'].set_ylim(0, 1600)
                self.axs['Distance'].legend()
        if len(indexes)>0:
            self.figs['Distance'].canvas.draw()
        return True

    def call_back2(self):
        data_distance = {}
        data_time = {}
        data_targets = {}
        for hand in ['left', 'right']:
            data_distance[hand] = []
            data_time[hand] = []
            data_targets[hand] = []
                        
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
                else:
                    print('plot_target not found')
        for hand in ['left', 'right']:
            if len(data_distance)>0:
                self.plot_metric(data_distance[hand], 'Distance', hand)
            if len(data_time)>0:
                self.plot_metric(data_time[hand], 'Time to impact', hand)
            if len(data_targets)>0:
                self.plot_metric(data_targets[hand], 'Targets', hand)
            
        if len(data_distance)>0 or len(data_time)>0 or len(data_targets)>0:
            self.fig.canvas.draw()
        # if len(data_distance)>0 or len(data_time)>0:
        return True

    def plot_metric(self,commands, metric, hand):
        # print(f'commands: {commands}')
        metric_hand = metric + hand
        if metric != 'Distance' and metric != 'Time to impact':
            # print('plot_target not found')
            return False
        if hand == 'left':
            i = 0
        else :
            i = 1
        if metric == 'Distance':
            j = 0
            ymin = 0
            ymax = 600
        elif metric == 'Time to impact':
            j = 1
            ymin = -2
            ymax = 2
        elif metric == 'Targets':
            j = 2
            ymin = 0
            ymax = 4
            
        all_data = {}
        all_timestamps = {}
        
        for command in commands:
            if command['label'] not in all_timestamps.keys():
                all_data[command['label']] = [command]
                all_timestamps[command['label']] = [command['time']]
            else:
                all_data[command['label']].append(command)
                all_timestamps[command['label']].append(command['time'])
              
        # find all inedexes of maximum timestamps
        indexes = {}
        for key in all_timestamps.keys():
            lab_timestamps = all_timestamps[key]
            if len(lab_timestamps)>0:
                max_timestamp = max(lab_timestamps)
                indexes [key]= [i for i, x in enumerate(lab_timestamps) if x == max_timestamp]
        # print(f'indexes: {indexes}')
        # print(f'all_data: {all_data}')
        # if len(indexes)>0:
        #     self.axs[i,j].clear()
            
        for key, data in all_data.items():
            for index in indexes[key]:
                command = data[index]
                x=command['x']
                y=command['y']
                # print(f'command: {command}')
                line_label = command['label']+command['plot_type']+metric_hand
                print(f'line_label: {line_label}')
                if line_label not in self.lines.keys():
                    self.lines[line_label] = self.axs[i,j].plot(x, y, command['plot_marker'], color = command['color'], label = line_label)[0]
                else:
                    self.lines[line_label].set_data(x, y)
                # self.axs[i,j].plot(x, y, command['plot_marker'], color = command['color'], label = command['label']+command['plot_type'])
                
                # if metric_hand not in self.plot_is_new.keys():
                #     self.plot_is_new[metric_hand] = False
                # else:
                #     # self.axs[i,j].set_xlabel('Timestamps')
                #     # self.axs[i,j].set_ylabel(hand)
                #     self.axs[i,j].set_xlim(-0.6,0.6)
                #     self.axs[i,j].set_ylim(ymin, ymax)
                    
                object_label = command['object_label']
                if object_label not in self.targets.keys():
                    self.targets[object_label] = command['color']
                    self.legend_handles.append(mlines.Line2D([], [], color=command['color'], marker='o', linestyle='None', label=object_label))
                    self.fig.legends=[]
                    self.fig.legend(handles=self.legend_handles, loc='right', bbox_to_anchor=(1., 0.5))
                # self.axs[i,j].legend()
                
            
    def plot_target_new(self,commands, target):
        all_data = {}
        all_timestamps = {}
        for command in commands:
            if command['label'] not in all_timestamps.keys():
                all_data[command['label']] = [command]
                all_timestamps[command['label']] = [command['time']]
            else:
                all_data[command['label']].append(command)
                all_timestamps[command['label']].append(command['time'])
              
        # find all inedexes of maximum timestamps
        indexes = {}
        for key in all_timestamps.keys():
            lab_timestamps = all_timestamps[key]
            if len(lab_timestamps)>0:
                max_timestamp = max(lab_timestamps)
                indexes [key]= [i for i, x in enumerate(lab_timestamps) if x == max_timestamp]
        # print(f'indexes: {indexes}')
        # print(f'all_data: {all_data}')
        if len(indexes)>0:
            self.axs[target].clear()
            
        for key, data in all_data.items():
            for index in indexes[key]:
                command = data[index]
                self.x=command['x']
                self.y=command['y']
                # print(f'command: {command}')
                self.axs[target].plot(self.x, self.y, command['plot_marker'], color = command['color'], label = command['label']+command['plot_type'])
                self.axs[target].legend()
            
    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.axs = plt.subplots(2,3)
        self.fig.subplots_adjust(right=0.8)
        self.fig.suptitle('Target detection')
        # self.fig.set_size_inches(18.5, 10.5)
        self.fig.set_size_inches(18.5/2, 10.5/2)
        hands= ['left', 'right']
        
        self.axs[0,0].set_title('Distance')
        self.axs[0,0].set_ylim(0, 1600)
        self.axs[0,0].set_xlim(-0.6,0.6)
                
        self.axs[0,1].set_title('Time to impact')
        self.axs[0,1].set_ylabel(hands[0])
        self.axs[0,1].set_ylim(-2, 2)
        self.axs[0,1].set_xlim(-0.6,0.6)
        
        self.axs[0,2].set_title('Targets')
        self.axs[0,2].set_ylim(0, 4)
        self.axs[0,2].set_xlim(-0.6,0.6)
        
        self.axs[1,0].set_xlabel('Timestamps')
        self.axs[1,1].set_xlabel('Timestamps')
        self.axs[1,2].set_xlabel('Timestamps')
        self.axs[1,0].sharex(self.axs[0,0])
        self.axs[1,1].sharex(self.axs[0,1])
        self.axs[1,2].sharex(self.axs[0,2])
        
        # self.axs[0,2].sharey(self.axs[0,0])
        
        self.axs[1,0].set_ylabel(hands[1])
        # self.axs[1,1].sharey(self.axs[1,0])
        # self.axs[1,2].sharey(self.axs[1,0])
        
        sampled_data_handle=mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='samples')
        interpolated_data_handle=mlines.Line2D([], [], color='black', linestyle='-',  label='interpolated')
        extrapolated_data_handle=mlines.Line2D([], [], color='black', linestyle='--',  label='extrapolated')
        self.legend_handles = [sampled_data_handle, interpolated_data_handle, extrapolated_data_handle]
        self.fig.legend(handles=self.legend_handles, loc='right', bbox_to_anchor=(1., 0.5))
        
        timer = self.fig.canvas.new_timer(interval=30)
        timer.add_callback(self.call_back2)
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