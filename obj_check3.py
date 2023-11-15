#!/usr/bin/env python3

import argparse
from i_grip import HandDetectors2 as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene as sc
import ExperimentReplayer as er
from i_grip.utils import kill_gpu_processes
import torch
import gc
import os
import cv2
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   print(torch.cuda.memory_snapshot())
   torch.cuda.empty_cache()

class GraspingDetector:
    def __init__(self, ) -> None:
        dataset = "ycbv"
        # self.hand_detector = hd.HybridOAKMediapipeDetector(detect_hands=False)
        cam_data = np.load('/home/emoullet/Documents/i-GRIP/DATA/Session_1/cam_19443010910F481300.npz')
        self.exp=er.ExperimentReplayer(0, cam_data, name = 'test', display_replay = True, resolution=(1280,720), fps=30.0)



    def run(self):
        self.exp.replay('j')

    def stop(self):
        self.scene.stop()
        cv2.destroyAllWindows()
        self.object_detector.stop()
        self.object_pose_estimator.stop()
        exit()


        

if __name__ == '__main__':
    
    # report_gpu()
    kill_gpu_processes()
    
    cam_data = np.load('/home/emoullet/Documents/i-GRIP/DATA/Session_1/cam_19443010910F481300.npz')
    exp=er.ExperimentReplayer(0, cam_data, name = 'test', display_replay = True, resolution=(1280,720), fps=30.0)
    print('test là')
    exp.replay('j')
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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    i_grip = GraspingDetector()
    i_grip.run()