#!/usr/bin/env python3

import argparse
from i_grip import HandDetectors2 as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene2 as sc
import cv2
import numpy as np
import os
import pandas as pd
import time

class ExperimentAnalyser:
    def __init__(self, device_id, device_data, name = None, display_replay = True, resolution=(1280,720), fps=60.0) -> None:

        
        self.display_replay = display_replay
        if name is None:
            self.name = f'ExperimentAnaliser'
        else:
            self.name = name
        self.scene = sc.AnalysiScene( device_data, name = f'{self.name}_scene', fps = fps)
        self.device_id = device_id
    
    def get_device_id(self):
        return self.device_id
    
    def analyse(self, hands_label, obj_labels, all_trajectories:pd, name = None, video_path = None):
        print(f'all_trajectories: {all_trajectories}')
        self.scene.reset()
        if video_path is not None:
            cap = cv2.VideoCapture(video_path)
            
        
        if name is not None:
            cv_window_name = f'{self.name} : Replaying {name}'
        else:
            cv_window_name = f'{self.name} : Replaying'
            
        hand_c = ['x', 'y', 'z']
        obj_c = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        
        hands = {}
        for hand_label in hands_label:
            df = pd.DataFrame(columns = ['Timestamps']+hand_c)
            df['Timestamps'] = all_trajectories['Timestamps']
            keep_label = True
            for c in hand_c:
                col_label = hand_label+'_hand_'+c
                if col_label not in all_trajectories.columns:
                    keep_label = False 
                    print(f' Hand {hand_label} has a file in the directory but its label was not found in the main trajectory file')
                else:
                    
                    df[c] = all_trajectories[col_label]
            if keep_label:
                hands[hand_label] = df
            
        expected_objects = sc.RigidObject.LABEL_EXPE_NAMES    
        # remove unexpected objects
        keys_to_remove = []
        for label in obj_labels:
            if label not in expected_objects:
                keys_to_remove.append(label)
        for key in keys_to_remove:
            obj_labels.remove(key)

        objects = {}
        for obj_label in obj_labels:
            df = pd.DataFrame(columns = ['Timestamps']+obj_c)
            df['Timestamps'] = all_trajectories['Timestamps']
            keep_label = True
            for c in obj_c:
                col_label = obj_label+'_'+c
                if col_label not in all_trajectories.columns:
                    keep_label = False
                    print(f' Object {obj_label} has a file in the directory but its label was not found in the main trajectory file')
                else:
                    df[c] = all_trajectories[col_label]
            if keep_label:
                objects[obj_label] = df
            
            
        # TODO : gérer les NaN dans les trajectoires
            # break
        print('START')
        print(f'Number of frames: {len(all_trajectories)}')
        for i, row in all_trajectories.iterrows():
            hands_labels_to_remove = []
            for hand_lab, traj in hands.items():
                #  check if there is a NaN in the ith row of the trajectory
                if not traj.iloc[i].isnull().values.any():
                    # remove the NaN from the trajectory
                    traj = traj.dropna()
                    self.scene.new_hand(hand_lab, traj)
                    hands_labels_to_remove.append(hand_lab)
                    
            for hand_lab in hands_labels_to_remove:
                del hands[hand_lab]
            
            objects_labels_to_remove = []
            for obj_lab, traj in objects.items():
                if not traj.iloc[i].isnull().values.any():
                    traj = traj.dropna()
                    self.scene.new_object(obj_lab, traj, dataset='ycbv')
                    objects_labels_to_remove.append(obj_lab)
            
            for obj_lab in objects_labels_to_remove:
                del objects[obj_lab]
            
            self.scene.next_timestamp(row['Timestamps'])
            # self.scene.draw(cv_window_name)
            print('NEXT')
            if video_path is not None:
                ret, img = cap.read()
                if not ret:
                    continue
                cv2.imshow(cv_window_name, img)
                cv2.imwrite(f'/home/emoullet/Documents/i-GRIP/DATA/Session_1_processing/Pre_processing/RKS6625/vids_frames/{self.name}_{i}.png', img)
                cv2.waitKey(1)
            # sleep 30ms
            time.sleep(0.05)
        target_data = self.scene.get_target_data()
        return target_data
        
    def stop(self):
        print("Stopping scene...")
        self.scene.stop()
        print("Stopped scene...")
        
        cv2.destroyAllWindows()
        

        

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
    i_grip = ExperimentReplayer(**args)
    i_grip.run()