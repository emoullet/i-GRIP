#!/usr/bin/env python3

import matplotlib
matplotlib.use('tkAgg')
import multiprocessing as mp
import matplotlib.pyplot as plt

class ProcessPlotter(object):
    def __init__(self):
        self.x = []
        self.y = []

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
        all_data = {}
        all_timestamps = {}
        data_distance = []
        data_time = []
            
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                if command['plot_target'] == 'Distance':
                    data_distance.append(command)
                elif command['plot_target'] == 'Time to impact':
                    data_time.append(command)
                else:
                    print('plot_target not found')
                    
        if len(data_distance)>0:
            self.plot_target(data_distance, 'Distance')
            self.figs['Distance'].canvas.draw()
        if len(data_time)>0:
            self.plot_target(data_time, 'Time to impact')
            self.figs['Time to impact'].canvas.draw()
        # if len(data_distance)>0 or len(data_time)>0:
        return True

    def plot_target(self,commands, target):
        if target != 'Distance' and target != 'Time to impact':
            print('plot_target not found')
            return False
        if target == 'Distance':
            ymin = 0
            ymax = 1600
        elif target == 'Time to impact':
            ymin = -2
            ymax = 2
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
                print(f'command: {command}')
                self.axs[target].plot(self.x, self.y, command['plot_marker'], color = command['color'], label = command['label']+command['plot_type'])
                self.axs[target].set_xlabel('Timestamps')
                self.axs[target].set_ylabel(target)
                self.axs[target].set_xlim(-0.6,0.6)
                self.axs[target].set_ylim(ymin, ymax)
                self.axs[target].legend()
            
    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        figs_labels = ['Distance', 'Time to impact', 'Targets']
        self.figs = {}
        self.axs = {}
        for label in figs_labels:
            self.figs[label], self.axs[label] = plt.subplots()
            self.figs[label].suptitle(label)
            timer = self.figs[label].canvas.new_timer(interval=20)
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