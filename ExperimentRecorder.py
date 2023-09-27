#!/usr/bin/env python3

import argparse
import threading
from i_grip import HandDetectors2 as hd
import cv2
import numpy as np
import time
import pandas as pd
import os

class ExperimentRecorder:
    def __init__(self, main_path, device_id = None, resolution=(1280,720), fps=30.0):
        print(f"Recorder created at {main_path}")
        self.main_path = main_path
        self.cam_label = device_id
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.hand_detector = hd.HybridOAKMediapipeDetector( device_id=device_id, detect_hands=False, resolution=resolution, fps=fps)
        self.device_data = self.hand_detector.get_device_data()
        res = self.device_data['resolution']
        # self.res = (res[0], res[1])
        self.path_cam_np = os.path.join(self.main_path, f'cam_{self.cam_label}_{res[0]}_{res[1]}_data.npz')
        if not os.path.exists(self.path_cam_np):
            np.savez(self.path_cam_np, **self.device_data)
        self.depth_map_series = []
        self.time_series = []
        self.recording = False
        print(f'Recorder with {device_id} built.')
        self.img = None
        obj_path = '.YCBV_test_pictures/javel.png'
        self.obj_img = None

    def init(self):
        self.hand_detector.start()
        self.capture_thread = threading.Thread(target=self.capture_task)
        self.record_thread = None
        self.current_path = None
        self.current_recording = None
        self.capture_thread.start()
        self.saving_threads=[]
        print(f'Recorder with {self.device_id} started.')

    def capture_task(self):
        self.new_rec=False
        self.end_rec = False
        while self.hand_detector.isOn():
            success, img = self.hand_detector.next_frame()
            if not success:
                continue
            map = self.hand_detector.get_depth_map()
            #cv2.imshow(f'view {self.device_id}',img)

            if self.new_rec:
                self.new_rec = False
                recorder = cv2.VideoWriter(self.path_vid, self.fourcc, self.fps, self.device_data['resolution'])
                self.recording=True
                

            self.img = img
            if self.obj_img is not None:
                self.img[:self.obj_img.shape[0], :self.obj_img.shape[1]] = self.obj_img
            if self.recording:
                t = pd.Timestamp.now()
                self.time_series.append(t)
                self.depth_map_series.append(map)
                recorder.write(self.img)
                if self.end_rec:
                    self.end_rec = False
                    recorder.release()
                    self.recording = False
    
    def save_data_task(self):
        #recorder = cv2.VideoWriter(self.path_vid, self.fourcc, 30.0,(1280,720))
        #for im in self.img_series:
        #    recorder.write(im)
        if self.current_path is None:
            print("No recording to save")
            return
        print(f"Start saving {self.current_path}")
        path= self.path_gzip
        path_timestamps = self.path_timestamps
        dm_series = self.depth_map_series
        t_series= self.time_series
        df = pd.DataFrame({'Depth_maps': dm_series,
                           'Date': t_series})
        df['Timestamps']= (df['Date']-df['Date'][0]).dt.total_seconds()
        t = time.time()
        df.to_pickle(path, compression='gzip')
        print('gzip compress time', time.time()-t)
        #extract timestamps into new dataframe
        new_df = pd.DataFrame()
        new_df['Timestamps'] = df['Timestamps']
        #save timestamps as csv
        new_df.to_pickle(path_timestamps, compression='gzip')
        print(f"Finished saving {path}")


    def new_record(self, name):
        print(f"Starting recording {self.device_id} with config {name}")
        self.current_recording = name
        self.time_series=[]
        self.depth_map_series=[]
        self.current_path= os.path.join(self.main_path, name)
        self.path_vid = os.path.join(self.current_path, f'{name}_cam_{self.cam_label}_video.avi')
        self.path_gzip = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_depth_map.gzip')
        self.path_timestamps = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_timestamps.csv')
        #self.recorder = cv2.VideoWriter(self.path_vid, self.fourcc, 30.0,(1280,720))
        self.new_rec = True

    def record_trial(self, trial):
        name = trial.label
        print(f"Starting recording {self.device_id} with config {name}")
        self.current_recording = name
        self.time_series=[]
        self.depth_map_series=[]
        self.current_path= os.path.join(self.main_path, name)
        self.path_vid = os.path.join(self.current_path, f'{name}_cam_{self.cam_label}_video.avi')
        self.path_gzip = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_depth_map.gzip')
        self.path_timestamps = os.path.join(self.current_path,f'{name}_cam_{self.cam_label}_timestamps.csv')
        self.new_rec = True
    
    def stop_record(self):
        self.end_rec = True
        if self.current_recording is None:
            print("No recording to stop")
            return
        print(f"Stoping recording {self.device_id} with config {self.current_recording}")
        save_thread = threading.Thread(target=self.save_data_task)
        save_thread.start()
        self.saving_threads.append(save_thread)
    
    def stop(self):
        print(f"Stoping {self.device_id}")
        self.hand_detector.stop()
        print("Waiting for threads to stop")
        self.capture_thread.join()
        print("Capture thread stopped")
        for thread in self.saving_threads:
            thread.join()
            print("Saving thread stopped")
        print(f"Stopped {self.device_id}")


        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-hd', '--hand_detection', choices=['mediapipe', 'depthai', 'hybridOAKMediapipe'],
                        default = 'hybridOAKMediapipe', help="Hand pose reconstruction solution")
    parser.add_argument('-od', '--object_detection', choices=['cosypose, megapose'],
                        default = 'cosypose', help="Object pose reconstruction detection")
    args = vars(parser.parse_args())

    # if args.hand_detection == 'mediapipe':
    #     import mediapipe as mp
    # else:
    #     import depthai as dai
    
    # if args.object_detection == 'cosypose':
    #     import cosypose
    grasp_int = ExperimentRecorder(**args)
    grasp_int.init()