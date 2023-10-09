#!/usr/bin/env python3

import argparse
from i_grip import HandDetectors2 as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene2 as sc
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ttkbootstrap import Style
import os
import pandas as pd
import threading

class ExperimentPreProcessor:
    def __init__(self, name = None) -> None:
        
        if name is None:
            self.name = f'ExperimentPreProcessor'
        else:
            self.name = name
        self.x, self.y, self.w, self.h = 150,20,300,400
        style = Style(theme="superhero")
        self.processing_window =style.master
        self.processing_window.title(f"Processing {name}")
        self.processing_window.geometry("400x200")
        self.image_label = ttk.Label(self.processing_window)
        self.image_label.pack()
        self.start_var = tk.DoubleVar()
        self.start_var.set(0)
        self.end_var = tk.DoubleVar()
        self.end_var.set(100)
        
        start_frame = ttk.Frame(self.processing_window)
        start_frame.pack(fill=tk.X, expand=True, padx=20)
        
        end_frame = ttk.Frame(self.processing_window)
        end_frame.pack(fill=tk.X, expand=True, padx=20)
        end_frame.columnconfigure(1, weight=1)
        
        self.start_label = ttk.Label(start_frame, text=f"Start : {self.start_var.get()}")
        self.start_label.grid(row=0, column=0, sticky=tk.W)
        start_frame.columnconfigure(1, weight=1)
        
        self.end_label = ttk.Label(end_frame, text=f"End : {self.end_var.get()}")
        self.end_label.grid(row=0, column=0, sticky=tk.W)
        
        self.start_trackbar = ttk.Scale(start_frame, variable=self.start_var, from_=0, to=100, orient=tk.HORIZONTAL, command=self.onChangeStart)
        self.start_trackbar.grid(row=0, column=1,  sticky='ew')
        
        self.end_trackbar = ttk.Scale(end_frame, variable=self.end_var, from_=0, to=100, orient=tk.HORIZONTAL, command=self.onChangeEnd)
        self.end_trackbar.grid(row=0, column=1,columnspan=3, sticky='ew')
        
        self.next_button = ttk.Button(self.processing_window, text="Next", command=self.next_trial)
        self.next_button.pack()
        self.play_button = ttk.Button(self.processing_window, text="Play", command=self.play)
        self.play_button.pack()
        self.cut_and_save_button = ttk.Button(self.processing_window, text="Cut and Save", command=self.cut_and_save)
        self.cut_and_save_button.pack()
        
        self.loop_var = tk.BooleanVar()
        self.bloop = False
        self.loop_button = ttk.Checkbutton(self.processing_window, text="Loop video play", command=self.loop, variable=self.loop_var)
        self.loop_button.pack()
        
        self.rotate_var = tk.BooleanVar()
        self.brotate = False
        self.rotate_button = ttk.Checkbutton(self.processing_window, text="Loop video play", command=self.rotate, variable=self.rotate_var)
        self.rotate_button.pack()
        
        self.face_visible = tk.BooleanVar()
        self.bface_visible = False
        self.face_visible_button = ttk.Checkbutton(self.processing_window, text="Face visible", command=self.face_visible, variable=self.face_visible)
        self.face_visible_button.pack()
        
        self.combination_respected = tk.BooleanVar()
        self.bcombination_respected = False
        self.combination_respected_button = ttk.Checkbutton(self.processing_window, text="Combination respected", command=self.combination_respected, variable=self.combination_respected)
        self.combination_respected_button.pack()
        
        self.fps = 30.0
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.resolution = (1280,720)
        self.save_threads = []
        
    
    def next_trial(self):
        folder_path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1/CID3008/trial_0_combi_bleach_left_pinch_simulated/'
        #list '.avi' files
        video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
        #get the path before '_video.avi'
        recording_paths = [os.path.join(folder_path, video_file).split('_video.avi')[0] for video_file in video_files]
        self.video_paths = [path + '_video.avi' for path in recording_paths]
        self.depthmap_paths = [path + '_depth_map.gzip' for path in recording_paths]
        self.timestamps_paths = [path + '_timestamps.csv' for path in recording_paths]
        self.mov_video_paths = [path + '_video_movement.avi' for path in recording_paths]
        self.mov_depthmap_paths = [path + '_depth_map_movement.gzip' for path in recording_paths]
        self.mov_timestamps_paths = [path + '_timestamps_movement.csv' for path in recording_paths]
        self.pre_process(recording_paths[0] + '_video.avi')


    def pre_process(self, replay, name = None):
        if isinstance(replay, dict):
            self.video = cv2.VideoCapture(replay['Video'])
            self.timestamps = replay['Timestamps']
            self.depth_maps = replay['Depth_maps']
        elif isinstance(replay, str):
            self.video = cv2.VideoCapture(replay)
            
        frame_width = int(self.video.get(3))  # Largeur de la frame
        frame_height = int(self.video.get(4))  # Hauteur de la frame
        print(f"frame_width : {frame_width}, frame_height : {frame_height}")
        self.nb_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start_trackbar.configure(to=self.nb_frames)
        self.end_trackbar.configure(to=self.nb_frames)
        self.current_frame_index = 0
        if not self.video.isOpened():
            print("Error reading video") 
            exit()
        
        if name is not None:
            self.cv_window_name = f'{self.name} : Pre-processing {name}'
        else:
            self.cv_window_name = f'{self.name} : Pre-processing'
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)
        self.onChangeEnd(self.nb_frames)
        self.onChangeStart(0)
        
        cv2.createTrackbar( 'start', self.cv_window_name, 0, self.nb_frames, self.onChange )
        cv2.setTrackbarPos('start', self.cv_window_name, 0)
        # cv2.createTrackbar( 'end'  , cv_window_name, 100, self.nb_frames, self.onChange )
        # cv2.waitKey()
        # start = cv2.getTrackbarPos('start', cv_window_name)
        # end = cv2.getTrackbarPos('end', cv_window_name)

        # if start >= end:
        #     raise Exception("start must be less than end")

        # self.video.set(cv2.CAP_PROP_POS_FRAMES,start)
        
        # while self.video.isOpened():
        #     err,img = self.video.read()
        #     if self.video.get(cv2.CAP_PROP_POS_FRAMES) >= end:
        #         break
        #     cv2.imshow(cv_window_name, img)
        #     k = cv2.waitKey(10) & 0xff
        #     if k==27:
        #         break
    
    def cut_and_save(self):
        sav_th = threading.Thread(target=self.cut_and_save_task)
        sav_th.start()
        self.save_threads.append(sav_th)
    
    def cut_and_save_task(self):
        start = int(self.start_var.get())
        end = int(self.end_var.get())
        
        for id, v_path in enumerate(self.video_paths):
            reader = cv2.VideoCapture(v_path)
            if self.brotate:
                res = (self.resolution[1], self.resolution[0])
            else:
                res = self.resolution
            recorder = cv2.VideoWriter(self.mov_video_paths[id], self.fourcc, self.fps, res)
            frame_index = 0
            while reader.isOpened():
                err,img = reader.read()
                if frame_index >= start and frame_index <= end:
                    if self.brotate:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    recorder.write(img)
                if frame_index > self.end:
                    break
                frame_index += 1
            recorder.release()
            reader.release()
            print(f'video {id} saved')
            
        for id, d_path in enumerate(self.depthmap_paths):
            df = pd.read_pickle(d_path, compression='gzip')
            df = df[start:end]
            df.to_pickle(self.mov_depthmap_paths[id], compression='gzip')
            print(f'depthmap {id} saved')
            
        for id, t_path in enumerate(self.timestamps_paths):
            df = pd.read_pickle(t_path, compression='gzip')
            df = df[start:end]
            df.to_pickle(self.mov_timestamps_paths[id], compression='gzip')
            print(f'timestamps {id} saved')
        
    
    def play(self):
        self.go_on = True
        self.start = int(self.start_var.get())
        self.end = int(self.end_var.get())
        for frame_index in range(self.start, self.end):
            print(f'play, frame_index : {frame_index}')
            self.video.set(cv2.CAP_PROP_POS_FRAMES,float(frame_index))
            err,img = self.video.read()
            self.to_display(img, frame_index)
            if not self.go_on:
                break
            #wait for 25 ms
        if self.bloop and self.go_on:
            self.play()
    
    def loop(self):
        self.bloop = not self.bloop
        print(f'loop : {self.bloop}')
        
    def rotate(self):
        self.brotate = not self.brotate
        print(f'rotate : {self.brotate}')
        
    def to_display(self, img, index = None):
        if img is None:
            return
        nimg = img.copy()
        print(f"rotate : {self.brotate}")
        # nimg = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if self.brotate:
            nimg = cv2.rotate(nimg, cv2.ROTATE_90_CLOCKWISE)
        if index is not None:            
            cv2.setTrackbarPos('start', self.cv_window_name, index)
        cv2.imshow(self.cv_window_name, nimg)
        k = cv2.waitKey(25)
        if k == 27:
            self.go_on = False
        # image = Image.fromarray(img)
        # image = image.resize((self.w, self.h), Image.LANCZOS)
        # self.pic = ImageTk.PhotoImage(image)
        # self.image_label.configure(image=self.pic)
        # self.image_label.image = self.pic
        # self.image_label.place(x=self.x, y=self.y)
    def onChangeStart(self, trackbarValue):
        print(f'onChangeStart, trackbarval : {trackbarValue}')
        st = float(trackbarValue)
        et = self.end_var.get()
        
        if st >= self.nb_frames:
            self.start_trackbar.set(self.nb_frames-1)
            st = self.nb_frames-1            
        
        if st >= et:
            self.end_trackbar.set(st+1)
            
        self.video.set(cv2.CAP_PROP_POS_FRAMES,float(trackbarValue))
        err,img = self.video.read()
        ind = int(float(trackbarValue))
        self.to_display(img, ind)
        self.start_var.set(ind)
        self.start_label.configure(text=f"Start : {ind}")
        
    def onChangeEnd(self, trackbarValue):
        print(f'onChangeEnd, trackbarval : {trackbarValue}')
        et = float(trackbarValue)
        st = self.start_var.get()
        
        if et <=1:
            self.end_trackbar.set(2)
            et = 2
            
        if st >= et :
            self.start_trackbar.set(et-1)
            
        self.video.set(cv2.CAP_PROP_POS_FRAMES,float(trackbarValue))
        err,img = self.video.read()
        self.to_display(img, int(float(trackbarValue)))
        ind = int(float(trackbarValue))
        self.end_var.set(ind)
        self.end_label.configure(text=f"End : {ind}")
    
    def onChange(self, trackbarValue):
        pass
    def run(self):
        self.processing_window.mainloop()
        

        

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
    i_grip = ExperimentPreProcessor('test')
    i_grip.run()