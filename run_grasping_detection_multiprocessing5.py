#!/usr/bin/env python3

import argparse
import multiprocessing
import torch
import gc
import os
import cv2

from i_grip import RgbdCameras as rgbd
from i_grip import Hands3DDetectors as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene_multiprocessing as sc
# from i_grip import Scene_ nocopy as sc
from i_grip import Plotters3 as pl
from i_grip.utils import kill_gpu_processes

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   print(torch.cuda.memory_snapshot())
   torch.cuda.empty_cache()

DATASET = 'ycbv'

def detect_hands_task( cam_data, stop_event, img_depth_queue, detected_hands_queue):
    hand_detector = hd.Hands3DDetector(cam_data,
                                            hd.Hands3DDetector.LIVE_STREAM_MODE)
    while True:
        if stop_event.is_set():
            break
        my_img, my_depth_map = img_depth_queue.get()
        # print('detect_hands_task: got img')
        # print(my_img.shape)
        detected_hands = hand_detector.get_hands(my_img, my_depth_map)
        if detected_hands is not None:
            # print('detect_hands_task: got hands')
            # print(detected_hands)
            detected_hands_queue.put(detected_hands)
            # print('detect_hands_task: sent hands')
        if not img_depth_queue.empty():
            img_depth_queue.get()
        # print('detect_hands_task: updated hands')
    hand_detector.stop()

def detect_objects_task(cam_data, stop_event, detect_event, img_queue, detected_objects_queue):
    object_detector = o2d.get_object_detector(DATASET, cam_data)
    while True:
        if stop_event.is_set():
            break
        detect_flag = detect_event.wait(0.5)
        if detect_flag:
            my_img = img_queue.get()
            # print('detect_objects_task: got img')
            # print(my_img.shape)
            detected_objects = object_detector.detect(my_img)
            if detected_objects is not None:
                # print('detect_objects_task: got objects')
                # print(detected_objects)
                detected_objects_queue.put(detected_objects)
                # print('detect_objects_task: sent objects')
                detect_event.clear()
        if not img_queue.empty():
            img_queue.get()
        # print('detect_objects_task: updated objects')
    object_detector.stop()
        
def estimate_objects_task(cam_data, stop_event, img_queue, object_detections_queue, estimated_objects_queue):
    object_pose_estimator = ope.get_pose_estimator(DATASET,
                                                        cam_data,
                                                        use_tracking = True,
                                                        fuse_detections=False)
    while True:
        if stop_event.is_set():
            break
        my_img = img_queue.get()
        # print('estimate_objects_task: got img')
        # print(my_img.shape)
        if not object_detections_queue.empty():
            my_object_detections = object_detections_queue.get()
        else:
            my_object_detections = None
        # print('estimate_objects_task: got objects')
        # print(my_object_detections)
        my_estimated_objects = object_pose_estimator.estimate(my_img, detections = my_object_detections)
        if my_estimated_objects is not None:
            # print('estimate_objects_task: got estimated objects')
            # print(my_estimated_objects)
            estimated_objects_queue.put(my_estimated_objects)
            # print('estimate_objects_task: sent estimated objects')
        if not img_queue.empty():
            img_queue.get()
        # print('estimate_objects_task: updated estimated objects')
        
    object_pose_estimator.stop()
        

def scene_analysis_task(cam_data, stop_event, detect_event, img_queue):
    plotter = pl.NBPlot()
    scene = sc.LiveScene(cam_data, name='Full tracking', plotter=plotter)
    j =0
    while True:
        j+=1
        if stop_event.is_set():
            break
        
        img = img_queue.get()          
        k = cv2.waitKey(1)          
        scene.render(img)
        cv2.imshow('render_img', img)
        if k == 32:
            print('DETEEEEEEEEEEEEEEEEEECT')
            detect_event.set()

        if k == 27:
            print('end')
            stop_event.set()
            break
        
class GraspingDetector:
    def __init__(self, ) -> None:
        self.dataset = "ycbv"
        
    
    def run(self):
        multiprocessing.set_start_method('spawn')
        dataset = DATASET
        rgbd_cam = rgbd.RgbdCamera()
        cam_data = rgbd_cam.get_device_data()
        
        
        plotter = pl.NBPlot()
        scene = sc.LiveScene(cam_data, name='Full tracking', plotter=plotter, dataset=dataset)
        
        stop_event = multiprocessing.Event()
        detect_event = multiprocessing.Event()
        
        input_queue_rgbd_frame_hands = multiprocessing.Queue(maxsize=1)
        input_queue_rgb_frame_object_detection = multiprocessing.Queue(maxsize=1)
        input_queue_rgb_frame_object_estimation = multiprocessing.Queue(maxsize=1)
        
        input_queue_hands = multiprocessing.Queue(maxsize=1)
        input_queue_object_detection = multiprocessing.Queue(maxsize=1)
        input_queue_object_estimation = multiprocessing.Queue(maxsize=1)
        
        process_hands_detection = multiprocessing.Process(target=detect_hands_task, 
                                                          args=(cam_data, stop_event, input_queue_rgbd_frame_hands, input_queue_hands,))
        
        process_object_detection = multiprocessing.Process(target=detect_objects_task, 
                                                           args=(cam_data, stop_event, detect_event, input_queue_rgb_frame_object_detection, input_queue_object_detection,))
        
        process_object_estimation = multiprocessing.Process(target=estimate_objects_task, 
                                                            args=(cam_data, stop_event, input_queue_rgb_frame_object_estimation, input_queue_object_detection, input_queue_object_estimation,))
        
        process_hands_detection.start()
        process_object_detection.start()
        process_object_estimation.start()
        rgbd_cam.start()
        
        obj_path = './YCBV_test_pictures/javel.png'
        obj_path2 = './YCBV_test_pictures/mustard_front.png'
        # obj_path = './YCBV_test_pictures/YCBV.png'
        obj_img = cv2.imread(obj_path)
        obj_img = cv2.resize(obj_img, (int(obj_img.shape[1]/2), int(obj_img.shape[0]/2)))
        
        obj_img2 = cv2.imread(obj_path2)
        obj_img2 = cv2.resize(obj_img2, (int(obj_img2.shape[1]/3), int(obj_img2.shape[0]/3)))
        
        detect_event.set()
        
        while rgbd_cam.is_on():
            success, img, depth_map = rgbd_cam.next_frame()
            if not success:
                continue
            
            # HANDS
            if not input_queue_rgbd_frame_hands.full():
                img_for_hands = img.copy()
                img_for_hands = cv2.cvtColor(img_for_hands, cv2.COLOR_RGB2BGR)
                img_for_hands.flags.writeable = False
                rgbd_frame = (img_for_hands, depth_map)
                input_queue_rgbd_frame_hands.put(rgbd_frame)
                # print(f'updated img for hands')
            
            img[0:obj_img.shape[0], 0:obj_img.shape[1]] = obj_img
            img[0:obj_img2.shape[0], img.shape[1]-obj_img2.shape[1]:] = obj_img2
            
            # OBJECTS
            if not input_queue_rgb_frame_object_detection.full():
                img_for_objects = img.copy()
                img_for_objects = cv2.cvtColor(img_for_objects, cv2.COLOR_RGB2BGR)
                img_for_objects.flags.writeable = False
                input_queue_rgb_frame_object_detection.put(img_for_objects)
                # print(f'updated img for objects')
            
            if not input_queue_rgb_frame_object_estimation.full():
                img_for_objects = img.copy()
                img_for_objects = cv2.cvtColor(img_for_objects, cv2.COLOR_RGB2BGR)
                img_for_objects.flags.writeable = False
                input_queue_rgb_frame_object_estimation.put(img_for_objects)
                # print(f'updated img for objects')
            
            # SCENE
            if not input_queue_hands.empty():
                detected_hands = input_queue_hands.get()
                # print(f'got hands')
                # print(detected_hands)
                scene.update_hands(detected_hands)
                # print(f'updated hands')
            if not input_queue_object_estimation.empty():
                estimated_objects = input_queue_object_estimation.get()
                # print(f'got estimated objects')
                # print(estimated_objects)
                scene.update_objects(estimated_objects)
                # print(f'updated estimated objects')
            k = cv2.waitKey(1)
            scene.render(img)
            cv2.imshow('render_img', img)
            if k == 27:
                print('end')
                break
        stop_event.set()
        process_hands_detection.join()
        process_object_detection.join()
        process_object_estimation.join()
        process_hands_detection.terminate()
        rgbd_cam.stop()
        exit()

    def run2(self):
        print(self.__dict__)
        print('start')
        self.hand_detector.start()
        start_event = multiprocessing.Event()
        detect_event = multiprocessing.Event()
        estimate_event = multiprocessing.Event()
        self.t_obj_d = multiprocessing.Process(target=self.detect_objects_task, args=(start_event, detect_event,estimate_event,))
        self.t_obj_e = multiprocessing.Process(target=self.estimate_objects_task, args=(start_event, estimate_event,))
        # self.t_plot = threading.Thread(target=self.plot_task)
        # self.t_plot.start()
        self.t_obj_d.start()
        self.t_obj_e.start()
        started = True
        obj_path = './YCBV_test_pictures/javel.png'
        # obj_path = './YCBV_test_pictures/mustard_front.png'
        # obj_path = './YCBV_test_pictures/YCBV.png'
        obj_img = cv2.imread(obj_path)
        obj_img = cv2.resize(obj_img, (int(obj_img.shape[1]/2), int(obj_img.shape[0]/2)))
        while self.hand_detector.isOn():
            # pl.plot()
            k = cv2.waitKey(2)
            success, img = self.hand_detector.next_frame()
            if not success:
                self.img_for_objects = None
                continue     
            else:
                img[0:obj_img.shape[0], 0:obj_img.shape[1]] = obj_img
                img_for_hands = img.copy()
                # img_for_hands = cv2.resize(img_for_hands, (int(self.hand_detector.resolution[0]/2), int(self.hand_detector.resolution[1]/2)))
                img_for_hands = cv2.cvtColor(img_for_hands, cv2.COLOR_RGB2BGR)
                img.flags.writeable = False
                if estimate_event.is_set() or detect_event.is_set():
                    self.img_for_objects = img.copy()
                    # incorporate obj_img in img_for_objects
                    self.img_for_objects = cv2.cvtColor(self.img_for_objects, cv2.COLOR_RGB2BGR)
                    self.img_for_objects.flags.writeable = False
                if started:
                    start_event.set()
                    detect_event.set()
                    started = False
                if not estimate_event.is_set():
                    estimate_event.set()
                hands = self.hand_detector.get_hands(img_for_hands)
                # print(f'img_for_hands.shape: {img_for_hands.shape}')    
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



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-hd', '--hand_detection', choices=['mediapipe', 'depthai', 'hybridOAKMediapipe'],
                        default = 'hybridOAKMediapipe', help="Hand pose reconstruction solution")
    parser.add_argument('-od', '--object_detection', choices=['cosypose, megapose'],
                        default = 'cosypose', help="Object pose reconstruction detection")
    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print('start')
    report_gpu()
    kill_gpu_processes()
    i_grip = GraspingDetector()
    i_grip.run()

