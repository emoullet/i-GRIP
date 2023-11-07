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
        self.saved_imgs = None
        if name is None:
            self.name = f'ExperimentPreProcessor'
        else:
            self.name = name
        self.x, self.y, self.w, self.h = 150,20,300,400
        style = Style(theme="superhero")
        # self.processing_window =style.master
        self.processing_window = tk.Toplevel()
        self.processing_window.title(f"{self.name} : Pre-processing")
        self.processing_window.geometry("1000x800")
        self.image_label = ttk.Label(self.processing_window)
        self.image_label.pack()
        
        self.participant_label = ttk.Label(self.processing_window, text=f"Participant : ")
        self.participant_label.pack(padx=10, pady=10)
        
        self.check_frame = ttk.LabelFrame(self.processing_window, text="Check trial")
        self.check_frame.pack()
        
        self.trial_label = ttk.Label(self.check_frame, text=f"Trial 0/0")
        self.trial_label.pack(padx=10, pady=10)
        
        
        self.combination_label = ttk.Label(self.check_frame, text=f"Combination {name}", justify=tk.CENTER)
        self.combination_label.pack(padx=10, pady=10)
        
        check_buttons_frame = ttk.Frame(self.check_frame)
        check_buttons_frame.pack(padx=10, pady=10)
        self.combination_respected = tk.BooleanVar()
        self.combination_respected.set(True)
        self.bcombination_respected = False
        self.combination_respected_button = ttk.Checkbutton(check_buttons_frame, text="Combination respected", variable=self.combination_respected)
        self.combination_respected_button.grid(row=0, column=0,padx=10, pady=10)
        
        self.face_visible = tk.BooleanVar()
        self.bface_visible = False
        self.face_visible_button = ttk.Checkbutton(check_buttons_frame, text="Face visible", variable=self.face_visible)
        self.face_visible_button.grid(row=0, column=1, padx=10, pady=10)
        
        
        self.cut_frame = ttk.LabelFrame(self.processing_window, text="Cut trial")
        self.cut_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        self.start_var = tk.DoubleVar()
        self.start_var.set(0)
        self.end_var = tk.DoubleVar()
        self.end_var.set(100)
        self.return_mov_start_var = tk.DoubleVar()
        self.return_mov_start_var.set(200)
        
        dur_lab = ttk.Label(self.cut_frame, text=f"Duration : ")
        dur_lab.pack(padx=10, pady=10)
        self.duration_label = ttk.Label(self.cut_frame, text=f"{self.end_var.get() - self.start_var.get()} frames")
        self.duration_label.pack(padx=10, pady=10)
        
        start_frame = ttk.Frame(self.cut_frame)
        start_frame.pack(fill=tk.X, expand=True, padx=20)
        start_frame.columnconfigure(1, weight=1)
        
        end_frame = ttk.Frame(self.cut_frame)
        end_frame.pack(fill=tk.X, expand=True, padx=20)
        end_frame.columnconfigure(1, weight=1)
        
        self.reach_start_label = ttk.Label(start_frame, text=f"Reaching start (frame index): {self.start_var.get()}")
        self.reach_start_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        self.reach_start_trackbar = ttk.Scale(start_frame, variable=self.start_var, from_=0, to=100, orient=tk.HORIZONTAL, command=self.onChangeStart)
        self.reach_start_trackbar.grid(row=0, column=1,  sticky='ew', padx=10, pady=10)
        
        self.reach_end_label = ttk.Label(end_frame, text=f"Reaching end (frame index): {self.end_var.get()}")
        self.reach_end_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)       
        self.reach_end_trackbar = ttk.Scale(end_frame, variable=self.end_var, from_=0, to=100, orient=tk.HORIZONTAL, command=self.onChangeEnd)
        self.reach_end_trackbar.grid(row=1, column=1,columnspan=3, sticky='ew', padx=10, pady=10)
        
        self.return_mov_start_label = ttk.Label(end_frame, text=f"Return start (frame index): {self.return_mov_start_var.get()}")
        self.return_mov_start_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)       
        self.return_mov_start_trackbar = ttk.Scale(end_frame, variable=self.return_mov_start_var, from_=0, to=100, orient=tk.HORIZONTAL, command=self.onChangeReturnMovStart)
        self.return_mov_start_trackbar.grid(row=2, column=1,columnspan=3, sticky='ew', padx=10, pady=10)
        
        play_frame = ttk.Frame(self.cut_frame)
        play_frame.pack(padx=10, pady=10)
        self.play_movement_button = ttk.Button(play_frame, text="Play movement", command=self.play_movement)
        self.play_movement_button.grid(row=0, column=0,  padx=10, pady=10)
        self.play_contact_button = ttk.Button(play_frame, text="Play contact", command=self.play_contact)
        self.play_contact_button.grid(row=0, column=1,  padx=10, pady=10)
        self.play_return_button = ttk.Button(play_frame, text="Play return", command=self.play_return)
        self.play_return_button.grid(row=0, column=2,  padx=10, pady=10)
        
        self.loop_var = tk.BooleanVar()
        self.bloop = False
        self.loop_button = ttk.Checkbutton(play_frame, text="Loop video play", command=self.loop, variable=self.loop_var)
        self.loop_button.grid(row=0, column=3, padx=10, pady=10)
        
        self.rotate_var = tk.BooleanVar()
        self.brotate = False
        self.rotate_button = ttk.Checkbutton(play_frame, text="Rotate video", command=self.rotate, variable=self.rotate_var)
        self.rotate_button.grid(row=0, column=4, padx=10, pady=10)
        
        self.cut_and_save_button = ttk.Button(self.cut_frame, text="Cut and Save", command=self.cut_and_save)
        self.cut_and_save_button.pack(padx=10, pady=10)
        
        
        self.next_button = ttk.Button(self.processing_window, text="Next", command=self.next_trial)
        self.next_button.pack(padx=10, pady=10)
        self.next_button.config(state='disabled')
        
        
        
        self.fps = 30.0
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.resolution = (1280,720)
        self.save_threads = []
        self.processing_window.update()
        self.durations = 0.
        
        self.cv_window_name = 'Pre-processing'
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar( 'start', self.cv_window_name, 0, 100, self.onChange )
        
    def next_trial(self):
        self.stay = False
        cv2.waitKey(50)
        self.current_trial_index += 1
        self.next_button.config(state='disabled')
        #dezactivate check buttons
        self.combination_respected.set(True)
        self.face_visible.set(False)
        self.processing_window.update()
    
    def set_new_participant(self, participant_name, nb_trials):
        self.participant_label.configure(text=f"Participant : {participant_name}")
        self.nb_trials = nb_trials
        self.current_trial_index = 0
    
    def process_trial(self, folder_path = None, combination = None, destination_folder = None):
        self.destination_folder = destination_folder
        folder = folder_path.split('/')[-1]
        
        self.check_frame.configure(text=f"Check trial {self.current_trial_index+1}/{self.nb_trials}")
        self.cut_frame.configure(text=f"Cut trial {self.current_trial_index+1}/{self.nb_trials}")
        self.trial_label.configure(text=folder)
        combi_txt = 'Combination : \n'
        for i in range(4):
            combi_txt += f'\n {combination.values[i]}'
        self.combination_label.configure(text=combi_txt)
        
        #list '.avi' filess
        self.video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
        print(f"video_files : {self.video_files}")
        recording_paths = [os.path.join(folder_path, video_file).split('_video.avi')[0] for video_file in self.video_files]
        
        self.video_paths = [path + '_video.avi' for path in recording_paths]
        self.depthmap_paths = [path + '_depth_map.gzip' for path in recording_paths]
        self.timestamps_paths = [path + '_timestamps.csv' for path in recording_paths]
        
        
        self.timestamps = pd.read_pickle(self.timestamps_paths[0], compression='gzip')
        
        print(f"video_paths : {self.video_paths}")
        print(f"depthmap_paths : {self.depthmap_paths}")
        print(f"timestamps_paths : {self.timestamps_paths}")
        self.pre_process(recording_paths)
        return self.combination_respected.get(), not self.face_visible.get(), self.durations

    def get_duration(self):        
        self.start = int(self.start_var.get())
        self.end = int(self.end_var.get())
        self.return_mov_start = int(self.return_mov_start_var.get())
        # get duration from timestamps
        
        self.durations = {}
        self.durations['stand'] = self.timestamps['Timestamps'].iloc[self.start] - self.timestamps['Timestamps'].iloc[0]
        self.durations['movement'] = self.timestamps['Timestamps'].iloc[self.end] - self.timestamps['Timestamps'].iloc[self.start]
        self.durations['contact'] = self.timestamps['Timestamps'].iloc[self.return_mov_start] - self.timestamps['Timestamps'].iloc[self.end]
        self.durations['return'] = self.timestamps['Timestamps'].iloc[-1] - self.timestamps['Timestamps'].iloc[self.return_mov_start]
        
        cut_timestamps = self.timestamps.iloc[self.start:self.end]
        cut_timestamps['Timestamps'] = cut_timestamps['Timestamps'] - cut_timestamps['Timestamps'].iloc[0]
        dur =''
        i = 0
        for d, t in self.durations.items(): 
            if i <3:
                dur += f"{d} : {t:.2f} s - "
            else:
                dur += f"{d} : {t:.2f} s"
            i+=1
        self.duration_label.configure(text=dur)
    
    def skip_trial(self):  
        self.current_trial_index += 1
    
    def pre_process(self, replays, name = None):
        self.videos = [cv2.VideoCapture(replay + '_video.avi') for replay in replays]
            
        frame_width = int(self.videos[0].get(3))  # Largeur de la frame
        frame_height = int(self.videos[0].get(4))  # Hauteur de la frame
        print(f"frame_width : {frame_width}, frame_height : {frame_height}")
        nbf = int(self.videos[0].get(cv2.CAP_PROP_FRAME_COUNT))
        self.nb_frames = min(nbf, len(self.timestamps))
        
        print(f"nb_frames : {self.nb_frames} (min between video [{nbf}] and timestamps[{len(self.timestamps)}])")
        
        self.reach_start_trackbar.configure(to=self.nb_frames-1)
        self.reach_end_trackbar.configure(to=self.nb_frames-1)
        self.return_mov_start_trackbar.configure(to=self.nb_frames-1)
        cv2.setTrackbarMax('start', self.cv_window_name, self.nb_frames-1)
        cv2.setTrackbarPos('start', self.cv_window_name, 0)
        
        self.start_var.set(0)        
        self.end_var.set(int(self.nb_frames/2))
        self.return_mov_start_var.set(self.nb_frames-1)
        self.get_duration()
        self.current_frame_index = 0
        for vid in self.videos:
            if not vid.isOpened():
                print("Error reading video") 
                exit()
        
        self.onChangeEnd(int(self.nb_frames/2))
        self.onChangeReturnMovStart(self.nb_frames-1)
        self.onChangeStart(0)
        
        self.stay=True
        while self.stay:
            cv2.waitKey(25)            
            self.processing_window.update()
    
    def cut_and_save(self):
        self.start = int(self.start_var.get())
        self.end = int(self.end_var.get())
        self.return_mov_start = int(self.return_mov_start_var.get())
        print("start cut and save thread")
        sav_th = threading.Thread(target=self.cut_and_save_task)
        sav_th.start()
        self.save_threads.append(sav_th)
        self.next_button.config(state='normal')
    
    def cut_and_save_task(self):        
        print('beginning cut and save')
        video_paths = self.video_paths.copy()   
        depthmap_paths = self.depthmap_paths.copy()
        timestamps_paths = self.timestamps_paths.copy()
        
        start = self.start
        end = self.end
        return_mov_start = self.return_mov_start
        
        brotate = self.brotate
        resolution = self.resolution
        fourcc = self.fourcc
        fps = self.fps
        nb_frames = self.nb_frames
        destination_paths = [os.path.join(self.destination_folder, video_file).split('_video.avi')[0] for video_file in self.video_files]
        
        stand_video_paths = [path + '_video_stand.avi' for path in destination_paths]
        stand_depthmap_paths = [path + '_depth_map_stand.gzip' for path in destination_paths]
        stand_timestamps_paths = [path + '_timestamps_stand.gzip' for path in destination_paths]
        
        mov_video_paths = [path + '_video_movement.avi' for path in destination_paths]
        mov_depthmap_paths = [path + '_depth_map_movement.gzip' for path in destination_paths]
        mov_timestamps_paths = [path + '_timestamps_movement.gzip' for path in destination_paths]
        
        contact_video_paths = [path + '_video_contact.avi' for path in destination_paths]
        contact_depthmap_paths = [path + '_depth_map_contact.gzip' for path in destination_paths]
        contact_timestamps_paths = [path + '_timestamps_contact.gzip' for path in destination_paths]
        
        ret_video_paths = [path + '_video_return.avi' for path in destination_paths]
        ret_depthmap_paths = [path + '_depth_map_return.gzip' for path in destination_paths]
        ret_timestamps_paths = [path + '_timestamps_return.gzip' for path in destination_paths]
        print("begin video saving")
        for id, v_path in enumerate(video_paths):
            reader = cv2.VideoCapture(v_path)
            if brotate:
                res = (resolution[1], resolution[0])
            else:
                res = resolution
            stand_recorder = cv2.VideoWriter(stand_video_paths[id], fourcc, fps, res)
            movement_recorder = cv2.VideoWriter(mov_video_paths[id], fourcc, fps, res)
            contact_recorder = cv2.VideoWriter(contact_video_paths[id], fourcc, fps, res)
            return_recorder = cv2.VideoWriter(ret_video_paths[id], fourcc, fps, res)
            frame_index = 0
            while reader.isOpened():
                err,img = reader.read()
                if brotate:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                if frame_index<start:
                    stand_recorder.write(img)                
                elif frame_index >= start and frame_index <= end:
                    movement_recorder.write(img)
                elif frame_index > end and frame_index <= return_mov_start:
                    contact_recorder.write(img)
                elif frame_index > return_mov_start:
                    return_recorder.write(img)
                if frame_index == nb_frames-1:                     
                    break
                frame_index += 1
            stand_recorder.release()
            movement_recorder.release()
            contact_recorder.release()
            return_recorder.release()
            reader.release()
            print(f'video {id} saved')
        print("begin depthmap saving")
        for id, d_path in enumerate(depthmap_paths):
            df = pd.read_pickle(d_path, compression='gzip')
            
            df_stand = df[:start]
            df_stand['Timestamps'] = df_stand['Timestamps'] - df_stand['Timestamps'].iloc[0]
            df_stand.to_pickle(stand_depthmap_paths[id], compression='gzip')
            
            df_mov = df[start:end]
            df_mov['Timestamps'] = df_mov['Timestamps'] - df_mov['Timestamps'].iloc[0]
            df_mov.to_pickle(mov_depthmap_paths[id], compression='gzip')
            
            df_con = df[end:return_mov_start]
            df_con['Timestamps'] = df_con['Timestamps'] - df_con['Timestamps'].iloc[0]
            df_con.to_pickle(contact_depthmap_paths[id], compression='gzip')
            
            df_ret = df[return_mov_start:]
            df_ret['Timestamps'] = df_ret['Timestamps'] - df_ret['Timestamps'].iloc[0]
            df_ret.to_pickle(ret_depthmap_paths[id], compression='gzip')
            
            print(f'depthmap {id} saved')
        print("begin timestamps saving")
        for id, t_path in enumerate(timestamps_paths):
            df = pd.read_pickle(t_path, compression='gzip')
            
            df_stand = df[:start]
            df_stand['Timestamps'] = df_stand['Timestamps'] - df_stand['Timestamps'].iloc[0]
            df_stand.to_pickle(stand_timestamps_paths[id], compression='gzip')
            
            df_mov = df[start:end]
            df_mov['Timestamps'] = df_mov['Timestamps'] - df_mov['Timestamps'].iloc[0]
            df_mov.to_pickle(mov_timestamps_paths[id], compression='gzip')
            
            df_con = df[end:return_mov_start]
            df_con['Timestamps'] = df_con['Timestamps'] - df_con['Timestamps'].iloc[0]
            df_con.to_pickle(contact_timestamps_paths[id], compression='gzip')
            
            df_ret = df[return_mov_start:]
            df_ret['Timestamps'] = df_ret['Timestamps'] - df_ret['Timestamps'].iloc[0]
            df_ret.to_pickle(ret_timestamps_paths[id], compression='gzip')
            
            print(f'timestamps {id} saved')
            movement_time = df_mov['Timestamps'].iloc[-1] - df_mov['Timestamps'].iloc[0]
        print("end cut and save")
        return movement_time
    
    def play_movement(self):
        self.start = int(self.start_var.get())
        self.end = int(self.end_var.get())
        self.play(self.start, self.end)
    
    def play_contact(self):
        self.end = int(self.end_var.get())
        self.return_mov_start = int(self.return_mov_start_var.get())
        self.play(self.end, self.return_mov_start)
        
    def play_return(self):
        self.return_mov_start = int(self.return_mov_start_var.get())
        self.play(self.return_mov_start, self.nb_frames-1)
        
    def play(self, start, end):
        self.go_on = True
        for frame_index in range(start, end):
            print(f'play, frame_index : {frame_index}')
            imgs = []
            for vid in self.videos:
                vid.set(cv2.CAP_PROP_POS_FRAMES,float(frame_index))
                err,img = vid.read()
                imgs.append(img)
            self.saved_imgs = imgs
            self.to_display(imgs, frame_index)
            if not self.go_on:
                break
        if self.bloop and self.go_on:
            self.play(start, end)
    
    def loop(self):
        self.bloop = not self.bloop
        print(f'loop : {self.bloop}')
        
    def rotate(self):
        self.brotate = not self.brotate
        print(f'rotate : {self.brotate}')
        self.to_display(self.saved_imgs)
        
    def to_display(self, imgs, index = None):
        nimgs=[]
        for img in imgs:
            if img is None:
                return
            # nimg = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
            nimg = img
            print(f"nimg.shape : {nimg.shape}")
            if self.brotate:
                nimg = cv2.rotate(nimg, cv2.ROTATE_90_CLOCKWISE)
            nimgs.append(nimg)
        if self.brotate:
            cimg = cv2.hconcat(nimgs)
        else:
            cimg = cv2.vconcat(nimgs)
        if index is not None:            
            cv2.setTrackbarPos('start', self.cv_window_name, index)
        cv2.imshow(self.cv_window_name, cimg)
        k = cv2.waitKey(25)
        if k == 27:
            self.go_on = False
        
    def onChangeStart(self, trackbarValue):
        # print(f'onChangeStart, trackbarval : {trackbarValue}')
        st = float(trackbarValue)
        et = self.end_var.get()
        rt = self.return_mov_start_trackbar.get()
        
        if st >= self.nb_frames-1:
            self.reach_start_trackbar.set(self.nb_frames-2)
            st = self.nb_frames-2
        
        if st >= et:
            self.reach_end_trackbar.set(st+1)
            
        ind = int(float(trackbarValue))
        imgs = []
        for vid in self.videos:
            vid.set(cv2.CAP_PROP_POS_FRAMES,float(trackbarValue))
            err,img = vid.read()
            imgs.append(img)
        self.to_display(imgs, ind)
        
        self.start_var.set(ind)
        if ind>=100:
            space=''
        elif ind>=10:
            space=' '
        else:
            space='  '
        self.reach_start_label.configure(text=f"Reaching start : {space}{ind}")
        self.saved_imgs = img
        self.get_duration()
        
    def onChangeEnd(self, trackbarValue):
        print(f'onChangeEnd, trackbarval : {trackbarValue}')
        et = float(trackbarValue)
        st = self.start_var.get()
        rt = self.return_mov_start_trackbar.get()
        
        if et <=1:
            self.reach_end_trackbar.set(2)
            self.return_mov_start_trackbar.set(2)
            et = 2
            
        if st >= et :
            self.reach_start_trackbar.set(et-1)
        
        if et >= rt :
            self.return_mov_start_trackbar.set(et+1)
            
        ind = int(float(trackbarValue))
        imgs = []
        for vid in self.videos:
            vid.set(cv2.CAP_PROP_POS_FRAMES,float(trackbarValue))
            err,img = vid.read()
            imgs.append(img)
        self.to_display(imgs, ind)
        
        self.end_var.set(ind)
        if ind>=100:
            space='  '
        elif ind>=10:
            space='   '
        else:
            space='    '
        self.reach_end_label.configure(text=f"Reaching end : {space}{ind}")
        self.saved_imgs = img
        self.get_duration()
        
    def onChangeReturnMovStart(self, trackbarValue):
        # print(f'onChangeReturnMovStart, trackbarval : {trackbarValue}')
        rt = float(trackbarValue)
        et = self.end_var.get()
        st = self.start_var.get()
        
        if rt <=1:
            self.reach_end_trackbar.set(2)
            self.reach_start_trackbar.set(2)
            rt = 2
            
        if et >= rt :
            self.reach_end_trackbar.set(rt-1)
            
        ind = int(float(trackbarValue))
        imgs = []
        for vid in self.videos:
            vid.set(cv2.CAP_PROP_POS_FRAMES,float(trackbarValue))
            err,img = vid.read()
            imgs.append(img)
        self.to_display(imgs, ind)
        
        self.return_mov_start_var.set(ind)
        if ind>=100:
            space='  '
        elif ind>=10:
            space='   '
        else:
            space='    '
        self.return_mov_start_label.configure(text=f"Return start : {space}{ind}")
        self.saved_imgs = img
        self.get_duration()
    
    
    def onChange(self, trackbarValue):
        pass
    def run(self):
        self.processing_window.mainloop()
    
    def stop(self):
        self.stay = False
        for th in self.save_threads:
            th.join()
        self.processing_window.destroy()

        

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