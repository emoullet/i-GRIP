#!/usr/bin/env python3

import argparse
import threading
import torch
import gc
import os
import cv2

from i_grip import HandDetectors2 as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene2 as sc
from i_grip import Plotters as pl
from i_grip.utils import kill_gpu_processes

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   print(torch.cuda.memory_snapshot())
   torch.cuda.empty_cache()


class GraspingDetector:
    def __init__(self, ) -> None:
        dataset = "ycbv"
        self.hand_detector = hd.HybridOAKMediapipeDetector()
        cam_data = self.hand_detector.get_device_data()
        plotter = pl.NBPlot()
        # plotter = None
        self.object_detector = o2d.get_object_detector(dataset,
                                                       cam_data)
        self.object_pose_estimator = ope.get_pose_estimator(dataset,
                                                            cam_data,
                                                            use_tracking = True,
                                                            fuse_detections=False)
        self.scene = sc.LiveScene(cam_data,
                              name = 'Full tracking', plotter = plotter)
        self.object_detections = None
        self.is_hands= False
        self.img_for_objects = None
        
        
    def estimate_objects_task(self, start_event, estimate_event):
        while self.hand_detector.isOn():
            start_flag = start_event.wait(1)
            if start_flag:
                if estimate_event.wait(1):
                    self.objects_pose = self.object_pose_estimator.estimate(self.img_for_objects, detections = self.object_detections)
                    self.scene.update_objects(self.objects_pose)
                    estimate_event.clear()

    def detect_objects_task(self, start_event, detect_event, estimate_event):
        while self.hand_detector.isOn():
            start_flag = start_event.wait(1)
            if start_flag:
                detect_flag = detect_event.wait(1)
                if detect_flag:
                    # self.object_detections = self.object_detector.detect(cv2.flip(self.img,1))
                    self.object_detections = self.object_detector.detect(self.img_for_objects)
                    if self.object_detections is not None:
                        detect_event.clear()
                        estimate_event.set()
                else:
                    self.object_detections = None


    def run(self):
        print(self.__dict__)
        print('start')
        self.hand_detector.start()
        start_event = threading.Event()
        detect_event = threading.Event()
        estimate_event = threading.Event()
        self.t_obj_d = threading.Thread(target=self.detect_objects_task, args=(start_event, detect_event,estimate_event,))
        self.t_obj_e = threading.Thread(target=self.estimate_objects_task, args=(start_event, estimate_event,))
        # self.t_plot = threading.Thread(target=self.plot_task)
        # self.t_plot.start()
        self.t_obj_d.start()
        self.t_obj_e.start()
        started = True
        obj_path = './YCBV_test_pictures/javel.png'
        obj_path = './YCBV_test_pictures/mustard_front.png'
        obj_path = './YCBV_test_pictures/YCBV.png'
        obj_img = cv2.imread(obj_path)
        while self.hand_detector.isOn():
            # pl.plot()
            k = cv2.waitKey(2)
            success, img = self.hand_detector.next_frame()
            if not success:
                self.img_for_objects = None
                continue     
            else:
                img_for_hands = img.copy()
                img_for_hands = cv2.cvtColor(img_for_hands, cv2.COLOR_RGB2BGR)
                img.flags.writeable = False
                if estimate_event.is_set() or detect_event.is_set():
                    self.img_for_objects = img.copy()
                    self.img_for_objects = cv2.cvtColor(self.img_for_objects, cv2.COLOR_RGB2BGR)
                    self.img_for_objects.flags.writeable = False
                if started:
                    start_event.set()
                    detect_event.set()
                    started = False
                if not estimate_event.is_set():
                    estimate_event.set()
                hands = self.hand_detector.get_hands(img_for_hands)
                if hands is not None and len(hands)>0:
                    self.scene.update_hands(hands)
                
                # Avant de commencer à utiliser la mémoire GPU
                torch.cuda.empty_cache()  # Pour libérer toute mémoire inutilisée

                # Utilisez cette ligne pour obtenir la mémoire GPU utilisée en octets
                gpu_memory_used = torch.cuda.memory_allocated()

                # Utilisez cette ligne pour obtenir la mémoire GPU réservée en octets (y compris la mémoire non allouée)
                gpu_memory_reserved = torch.cuda.memory_reserved()

                # Convertissez les valeurs en méga-octets (Mo) pour une meilleure lisibilité
                gpu_memory_used_mb = gpu_memory_used / 1024 / 1024
                gpu_memory_reserved_mb = gpu_memory_reserved / 1024 / 1024


                # print(f"GPU Memory Used: {gpu_memory_used_mb:.2f} MB")
                # print(f"GPU Memory Reserved: {gpu_memory_reserved_mb:.2f} MB")
                
            if k == 32:
                print('DETEEEEEEEEEEEEEEEEEECT')
                detect_event.set()
            self.scene.render(img)
            cv2.imshow('render_img', img)
            if k == 27:
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

    # return ax
    # return ax

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
    
    print('start')
    report_gpu()
    kill_gpu_processes()
    i_grip = GraspingDetector()
    i_grip.run()