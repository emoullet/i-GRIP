#!/usr/bin/env python3

import argparse
import multiprocessing
import torch
import gc
import os
import cv2
import tracemalloc
import time

from i_grip import RgbdCameras as rgbd
from i_grip import Hands3DDetectors as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
# from i_grip import Scene_refactored as sc
# from i_grip import Scene_refactored_multi as sc
# from i_grip import Scene_refactored_multi_thread_exec as sc
# from i_grip import Scene_refactored_multi_thread as sc
from i_grip import Scene_refactored_multi_fullthread as sc
# from i_grip import Scene_ nocopy as sc
from i_grip import Plotters3 as pl
# from i_grip import Plotters_queue as pl
from i_grip.utils import kill_gpu_processes

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   print(torch.cuda.memory_snapshot())
   torch.cuda.empty_cache()


def detect_hands_task( cam_data,hands, stop_event, img_depth_queue, detected_hands_queue):
    hand_detector = hd.Hands3DDetector(cam_data, hands = hands, running_mode =
                                            hd.Hands3DDetector.VIDEO_FILE_MODE, use_gpu=True)
    print('detect_hands_task: started')
    while True:
        if stop_event.is_set():
            break
        print('detect_hands_task: waiting for img')
        my_img, my_depth_map = img_depth_queue.get()
        # print('detect_hands_task: got img')
        # print(my_img)
        t = time.time()
        detected_hands = hand_detector.get_hands(my_img, my_depth_map,time.time())
        print('detect_hands_task: updated hands')
        print(f'detect_hands_task: {(time.time()-t)*1000:.2f} ms')
        if detected_hands is not None:
            # print('detect_hands_task: got hands')
            # print(detected_hands)
            detected_hands_queue.put(detected_hands)
            # print('detect_hands_task: sent hands')
        if not img_depth_queue.empty():
            img_depth_queue.get()
    hand_detector.stop()

def detect_objects_task(dataset, cam_data, stop_event, detect_event, img_queue, detected_objects_queue):
    object_detector = o2d.get_object_detector(dataset, cam_data)
    while True:
        t = time.time()
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
        print(f'detect_objects_task: {(time.time()-t)*1000:.2f} ms')
    object_detector.stop()
        
def estimate_objects_task(dataset, cam_data, stop_event, img_queue, object_detections_queue, estimated_objects_queue):
    object_pose_estimator = ope.get_pose_estimator(dataset,
                                                        cam_data,
                                                        use_tracking = True,
                                                        fuse_detections=False)
    while True:
        t = time.time()
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
        print(f'estimate_objects_task: {(time.time()-t)*1000:.2f} ms')
        
    object_pose_estimator.stop()
        

def scene_analysis_task(cam_data, stop_event, detect_event, img_queue, hands_queue, object_estimation_queue):
    plotter = pl.NBPlot()
    scene = sc.LiveScene(cam_data, name='Full tracking', plotter=plotter, )
    while True:
        # HANDS
        t_s = time.time()
        t = time.time()
        if not hands_queue.empty():
            estimated_hands = hands_queue.get()
        else:
            estimated_hands = None
            # print(f'got hands')
            # print(detected_hands)
        if estimated_hands is not None:
            scene.update_hands(estimated_hands)
            print(f'scene update hands: {(time.time()-t)*1000:.2f} ms')
                # print(f'updated hands')
        
        # OBJECTS
        t = time.time()
        estimated_objects = None
        # print('waiting for estimated objects')
        # print(out_object_estimation.poll())
        if not object_estimation_queue.empty():
            estimated_objects = object_estimation_queue.get()
        #     print(f'got estimated objects')
        #     print(estimated_objects)
        # print('finished waiting for estimated objects')
        if estimated_objects is not None:
            scene.update_objects(estimated_objects)
            print(f'scene update objects: {(time.time()-t)*1000:.2f} ms')
            # print(f'updated estimated objects')
        
        # IMAGE
        k = cv2.waitKey(1)
        t = time.time()
        img = None
        if not img_queue.empty():
            img = img_queue.get()
            # print(f'got img')
            # print(img.shape)
        print(f'get_queue time : {(time.time()-t)*1000:.2f} ms')
        if img is not None:
            t = time.time()
            scene.render(img)
            print(f'scene render: {(time.time()-t)*1000:.2f} ms')
            cv2.imshow('render_img', img)
            print(f'updated img : {(time.time()-t)*1000:.2f} ms')
        if k == 27:
            print('end')
            break
        print(f'scene analysis task: {(time.time()-t_s)*1000:.2f} ms')
    stop_event.set()
        
class GraspingDetector:
    def __init__(self, hands, dataset, fps, images) -> None:
        if hands == 'both':
            self.hands = ['left', 'right']
        else:
            self.hands = [hands]
        self.dataset = dataset
        self.fps = fps
        self.obj_images = images
    
    def run(self):
        tracemalloc.start()
        multiprocessing.set_start_method('spawn', force=True)
        rgbd_cam = rgbd.RgbdCamera(fps=self.fps)
        cam_data = rgbd_cam.get_device_data()
        
        stop_event = multiprocessing.Event()
        detect_event = multiprocessing.Event()
        
        
        queue_rgbd_frame_hands = multiprocessing.Queue(maxsize=1)
        queue_rgb_frame_object_detection = multiprocessing.Queue(maxsize=1)
        queue_rgb_frame_object_estimation = multiprocessing.Queue(maxsize=1)
        queue_rgb_frame_scene_analysis = multiprocessing.Queue(maxsize=1)
        
        queue_hands = multiprocessing.Queue(maxsize=1)
        queue_object_detection = multiprocessing.Queue(maxsize=1)
        queue_object_estimation = multiprocessing.Queue(maxsize=1)
        
        process_hands_detection = multiprocessing.Process(target=detect_hands_task, 
                                                          args=(cam_data, self.hands, stop_event, queue_rgbd_frame_hands, queue_hands,))
        
        process_object_detection = multiprocessing.Process(target=detect_objects_task, 
                                                           args=(self.dataset, cam_data, stop_event, detect_event, queue_rgb_frame_object_detection, queue_object_detection,))
        
        process_object_estimation = multiprocessing.Process(target=estimate_objects_task, 
                                                            args=(self.dataset,cam_data, stop_event, queue_rgb_frame_object_estimation, queue_object_detection, queue_object_estimation,))
        
        process_scene_analysis = multiprocessing.Process(target=scene_analysis_task, 
                                                        args=(cam_data, stop_event, detect_event, queue_rgb_frame_scene_analysis,queue_hands, queue_object_estimation))
        
        process_hands_detection.start()
        process_object_detection.start()
        process_object_estimation.start()
        process_scene_analysis.start()
        rgbd_cam.start()
        
        obj_imgs = []
        for img in self.obj_images:
            obj_img = cv2.imread(img)
            obj_img = cv2.resize(obj_img, (int(obj_img.shape[1]/2), int(obj_img.shape[0]/2)))
            obj_imgs.append(obj_img)
        
        detect_event.set()
        
        while rgbd_cam.is_on():
            t = time.time()
            success, img, depth_map = rgbd_cam.next_frame()
            print(f'frame collection time : {(time.time()-t)*1000:.2f} ms')
            if not success:
                continue
            
            t2 = time.time()
            
            # HANDS
            if not queue_rgbd_frame_hands.full():
                img_for_hands = img.copy()
                img_for_hands = cv2.cvtColor(img_for_hands, cv2.COLOR_RGB2BGR)
                img_for_hands.flags.writeable = False
                rgbd_frame = (img_for_hands, depth_map)
                queue_rgbd_frame_hands.put(rgbd_frame)
            
            # OBJECTS INSERTION
            for i, obj_img in enumerate(obj_imgs):
                if i == 0:
                    img[0:obj_img.shape[0], 0:obj_img.shape[1]] = obj_img
                elif i == 1:
                    img[0:obj_img.shape[0], img.shape[1]-obj_img.shape[1]:] = obj_img
                elif i == 2:
                    img[img.shape[0]-obj_img.shape[0]:, 0:obj_img.shape[1]] = obj_img
                elif i == 3:
                    img[img.shape[0]-obj_img.shape[0]:, img.shape[1]-obj_img.shape[1]:] = obj_img
            
            # OBJECT DETECTION
            if detect_event.is_set():                
                if not queue_rgb_frame_object_detection.full():
                    img_for_objects = img.copy()
                    img_for_objects = cv2.cvtColor(img_for_objects, cv2.COLOR_RGB2BGR)
                    img_for_objects.flags.writeable = False
                    queue_rgb_frame_object_detection.put(img_for_objects)

            # OBJECT ESTIMATION
            if not queue_rgb_frame_object_estimation.full():
                img_for_objects = img.copy()
                img_for_objects = cv2.cvtColor(img_for_objects, cv2.COLOR_RGB2BGR)
                img_for_objects.flags.writeable = False
                queue_rgb_frame_object_estimation.put(img_for_objects)
                # print(f'updated img for objects')
            
            # SCENE
            img_for_scene = img.copy()
            t = time.time()
            if not queue_rgb_frame_scene_analysis.full():
                print(f'polling time : {(time.time()-t)*1000:.2f} ms')
                t= time.time()
                queue_rgb_frame_scene_analysis.put( img_for_scene)
                print(f'sending time : {(time.time()-t)*1000:.2f} ms')
                # print(f'updated img for scene')
            
            print(f'frame sending time : {(time.time()-t2)*1000:.2f} ms')
            print('-------------------')
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.stop()
        process_hands_detection.join()
        process_object_detection.join()
        process_object_estimation.join()
        process_scene_analysis.join()
        rgbd_cam.stop()
        exit()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ha', '--hands', choices=['left', 'right', 'both'],
                        default = 'both', help="Hands to analyse for grasping intention detection")
    parser.add_argument('-d', '--dataset', choices=['t_less, ycbv'],
                        default = 'ycbv', help="Cosypose dataset to use for object detection and pose estimation")
    parser.add_argument('-f', '--fps', type=int, default=40, help="Frames per second for the camera")
    # parser.add_argument('-i', '--images', nargs='+', help="Path to the image(s) to use for object detection", default=['./YCBV_test_pictures/javel.png'])
    parser.add_argument('-i', '--images', nargs='+', help="Path to the image(s) to use for object detection", default=['./YCBV_test_pictures/javel.png', './YCBV_test_pictures/mustard_front.png'])
    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print('start')
    report_gpu()
    kill_gpu_processes()
    i_grip = GraspingDetector(**args)
    i_grip.run()

