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

class ExperimentAnalyser:
    def __init__(self, device_id, device_data, name = None, display_replay = True, resolution=(1280,720), fps=30.0) -> None:

        
        if name is None:
            self.name = f'ExperimentAnaliser'
        else:
            self.name = name
        self.scene = sc.AnalysiScene( device_data, name = f'{self.name}_scene')
        self.hand_detector.start()
    
    def get_device_id(self):
        return self.device_id
    
    def analyse(self, hands_label, obj_labels, all_trajectories:pd, name = None):
        
        self.scene.reset()
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
            for c in hand_c:
                df[c] = all_trajectories[hand_label+'_'+c]
            hands[hand_label] = df
        objects = {}
        for obj_label in obj_labels:
            df = pd.DataFrame(columns = ['Timestamps']+obj_c)
            df['Timestamps'] = all_trajectories['Timestamps']
            for c in obj_c:
                df[c] = all_trajectories[obj_label+'_'+c]
            objects[obj_label] = df
        for hand_lab, traj in hands.items():
            self.scene.new_hand(hand_lab, traj)
        for obj_lab, traj in objects.items():
            self.scene.new_object(obj_lab, traj)
        for i in range(len(all_trajectories)):
            self.scene.update()
            self.scene.draw(cv_window_name)
            cv2.waitKey(1)
        return hands_data, objects_data
        
    def stop(self):
        print("Stopping experiment replayer...")
        print("Stopping hand detector...")
        self.hand_detector.stop()
        print("Stopped hand detector...")
        print("Stopping object estimator...")
        self.object_pose_estimator.stop()
        print("Stopped object estimator...")
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