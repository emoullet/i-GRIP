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
from i_grip import Scene_refactored_multi_thread as sc
# from i_grip import Scene_refactored_multi_fullthread as sc
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
                                            hd.Hands3DDetector.LIVE_STREAM_MODE)
    print('detect_hands_task: started')
    while True:
        if stop_event.is_set():
            break
        print('detect_hands_task: waiting for img')
        my_img, my_depth_map = img_depth_queue.get()
        # print('detect_hands_task: got img')
        # print(my_img)
        t = time.time()
        detected_hands = hand_detector.get_hands(my_img, my_depth_map)
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
        hand_detector = hd.Hands3DDetector(cam_data, hands = self.hands, running_mode =
                                            hd.Hands3DDetector.VIDEO_FILE_MODE, use_gpu=True)
        
        scene = sc.LiveScene(cam_data, name='Full tracking', plotter=None, )
        rgbd_cam.start()
        
        while rgbd_cam.is_on():
            t = time.time()
            success, img, depth_map = rgbd_cam.next_frame()
            print(f'frame collection time : {(time.time()-t)*1000:.2f} ms')
            if not success:
                continue
            
            t2 = time.time()
            
            # HANDS
            img_for_hands = img.copy()
            img_for_hands = cv2.cvtColor(img_for_hands, cv2.COLOR_RGB2BGR)
            img_for_hands.flags.writeable = False
            rgbd_frame = (img_for_hands, depth_map)
            
            
            # print('detect_hands_task: got img')
            # print(my_img)
            t = time.time()
            estimated_hands = hand_detector.get_hands(*rgbd_frame, time.time())
            print('detect_hands_task: updated hands')
            print(f'detect_hands_task: {(time.time()-t)*1000:.2f} ms')
            # HANDS
            t_s = time.time()
            if estimated_hands is not None:
                scene.update_hands(estimated_hands)
                print(f'scene update hands: {(time.time()-t)*1000:.2f} ms')
                    # print(f'updated hands')
            
            # IMAGE
            k = cv2.waitKey(1)
            t = time.time()
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
            
            
        rgbd_cam.stop()
        exit()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ha', '--hands', choices=['left', 'right', 'both'],
                        default = 'both', help="Hands to analyse for grasping intention detection")
    parser.add_argument('-d', '--dataset', choices=['t_less, ycbv'],
                        default = 'ycbv', help="Cosypose dataset to use for object detection and pose estimation")
    parser.add_argument('-f', '--fps', type=int, default=30, help="Frames per second for the camera")
    # parser.add_argument('-i', '--images', nargs='+', help="Path to the image(s) to use for object detection", default=['./YCBV_test_pictures/javel.png'])
    parser.add_argument('-i', '--images', nargs='+', help="Path to the image(s) to use for object detection", default=['./YCBV_test_pictures/javel.png', './YCBV_test_pictures/mustard_front.png'])
    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print('start')
    report_gpu()
    kill_gpu_processes()
    i_grip = GraspingDetector(**args)
    i_grip.run()

