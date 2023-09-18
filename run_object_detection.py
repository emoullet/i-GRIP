#!/usr/bin/env python3

import argparse
import threading
from i_grip import HandDetectors as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene as sc
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
        dataset = "tless"
        self.hand_detector = hd.HybridOAKMediapipeDetector(detect_hands=False)
        cam_data = self.hand_detector.get_device_data()
        self.object_detector = o2d.get_object_detector(dataset,
                                                       cam_data)
        self.object_pose_estimator = ope.get_pose_estimator(dataset,
                                                            cam_data,
                                                            use_tracking = True,
                                                            fuse_detections=False)
        self.scene = sc.LiveScene(cam_data,
                              name = 'Full tracking')
        self.object_detections = None
        self.is_hands= False

    def estimate_objects_task(self, start_event, estimate_event):
        while self.hand_detector.isOn():
            start_flag = start_event.wait(1)
            if start_flag:
                if estimate_event.wait(1):
                    self.objects_pose = self.object_pose_estimator.estimate(self.rgb_img, detections = self.object_detections)
                    self.scene.update_objects(self.objects_pose)

    def detect_objects_task(self, start_event, detect_event, estimate_event):
        while self.hand_detector.isOn():
            start_flag = start_event.wait(1)
            if start_flag:
                detect_flag = detect_event.wait(1)
                if detect_flag:
                    # self.object_detections = self.object_detector.detect(cv2.flip(self.img,1))
                    self.object_detections = self.object_detector.detect(self.rgb_img)
                    if self.object_detections is not None:
                        detect_event.clear()
                        estimate_event.set()
                else:
                    self.object_detections = None


    def run(self):
        print(self.__dict__)
        self.hand_detector.start()
        print('start')
        start_event = threading.Event()
        detect_event = threading.Event()
        estimate_event = threading.Event()
        self.t_obj_d = threading.Thread(target=self.detect_objects_task, args=(start_event, detect_event,estimate_event,))
        self.t_obj_e = threading.Thread(target=self.estimate_objects_task, args=(start_event, estimate_event,))
        self.t_obj_d.start()
        self.t_obj_e.start()
        started = True
        obj_path = './YCBV_test_pictures/mustard_back.png'
        obj_path = './YCBV_test_pictures/YCBV2.png'
        # obj_path = './YCBV_test_pictures/YCBV.png'
        obj_img = cv2.imread(obj_path)
        # obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)
        while self.hand_detector.isOn():
            success, img = self.hand_detector.next_frame()
            if not success:
                self.img = None
                continue     
            else:
                render_img = img.copy()
                self.img = img.copy()
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                #replace pixels from self.img with obj_img
                # self.img = img
                rgb_img = cv2.cvtColor(obj_img.copy(), cv2.COLOR_BGR2RGB)
                # rgb_img = obj_img.copy()
                self.img = img.copy()
                self.rgb_img = img.copy()
                if started:
                    start_event.set()
                    detect_event.set()
                    estimate_event.set()
                    started = False
                estimate_event.set()                
                estimate_event.clear()
                
                # self.img = img
                # print('img shape', self.img.dtype)
                self.img[:obj_img.shape[0], :obj_img.shape[1]] = obj_img
                self.rgb_img[:obj_img.shape[0], :obj_img.shape[1]] = rgb_img
                self.rgb_img.flags.writeable = False
                cv2.imshow('rgb_img', self.rgb_img)
                # v2.cvtColor(self.img, cv2.COLOR_BGR2RGB, self.img)
            self.img.flags.writeable = True
            k = cv2.waitKey(1)
            # if k == 32:
            #     print('DOOOOOOOOOOOOOOOOOOOO')
            #     print('DOOOOOOOOOOOOOOOOOOOO')
            #     print('DOOOOOOOOOOOOOOOOOOOO')
            #     print('DOOOOOOOOOOOOOOOOOOOO')
            #     print('DOOOOOOOOOOOOOOOOOOOO')
            #     print('DOOOOOOOOOOOOOOOOOOOO')
            #     print('DOOOOOOOOOOOOOOOOOOOO')
            #     print('DOOOOOOOOOOOOOOOOOOOO')
            #     print('DOOOOOOOOOOOOOOOOOOOO')
            #     start_event.set()
            if k == 32:
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                print('DETEEEEEEEEEEEEEEEEEECT')
                detect_event.set()
                estimate_event.set()
            self.scene.render(render_img)
            cv2.imshow('image', render_img)
            if k==27:
                print('end')
                self.stop()
                break
        exit()

    def stop(self):
        self.hand_detector.stop()
        self.t_obj_d.join()
        self.t_obj_e.join()
        self.scene.stop()
        cv2.destroyAllWindows()
        self.object_detector.stop()
        self.object_pose_estimator.stop()
        exit()


        

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    report_gpu()
    kill_gpu_processes()
    i_grip = GraspingDetector()
    i_grip.run()