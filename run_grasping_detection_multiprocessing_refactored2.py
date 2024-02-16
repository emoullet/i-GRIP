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
from i_grip import Scene_refactored_multi as sc
# from i_grip import Scene_ nocopy as sc
from i_grip import Plotters3 as pl
from i_grip.utils import kill_gpu_processes

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   print(torch.cuda.memory_snapshot())
   torch.cuda.empty_cache()


def detect_hands_task( cam_data,hands, stop_event, img_depth_pipe, detected_hands_pipe):
    hand_detector = hd.Hands3DDetector(cam_data, hands = hands, running_mode =
                                            hd.Hands3DDetector.LIVE_STREAM_MODE)
    while True:
        t = time.time()
        if stop_event.is_set():
            break
        
        if img_depth_pipe.poll():
            # print('detect_hands_task: got img')
            while img_depth_pipe.poll():
                input = img_depth_pipe.recv()
                my_img = input['img']
                my_depth_map = input['depth_map']
        else:
            # print('detect_hands_task: didnt get img')
            input = img_depth_pipe.recv()
            # print('finally got img')
            my_img = input['img']
            my_depth_map = input['depth_map']
        # print('detect_hands_task: got img')
        # print(my_img)
        detected_hands = hand_detector.get_hands(my_img, my_depth_map)
        if detected_hands is not None:
            # print('detect_hands_task: got hands')
            # print(detected_hands)
            output = {'hands': detected_hands}
            detected_hands_pipe.send(output)
            # print('detect_hands_task: sent hands')
        # print('detect_hands_task: updated hands')
        print(f'detect_hands_task: {(time.time()-t)*1000:.2f} ms')
    hand_detector.stop()

def detect_objects_task(cam_data, stop_event, detect_event, img_pipe, detected_objects_pipe):
    object_detector = o2d.get_object_detector("ycbv", cam_data)
    while True:
        t = time.time()
        if stop_event.is_set():
            break
        detect_flag = detect_event.wait(0.5)
        if detect_flag:
            if img_pipe.poll():
                while img_pipe.poll():
                    # print('detect_objects_task: got img')
                    my_img = img_pipe.recv()['img']
            else:
                my_img = img_pipe.recv()['img']
            # print('detect_objects_task: got img')
            # print(my_img.shape)
            detected_objects = object_detector.detect(my_img)
            if detected_objects is not None:
                # print('detect_objects_task: got objects')
                # print(detected_objects)
                detected_objects_pipe.send({'detected_objects': detected_objects})
                # print('detect_objects_task: sent detected objects')
                detect_event.clear()
        # print('detect_objects_task: updated objects')
        print(f'detect_objects_task: {(time.time()-t)*1000:.2f} ms')
    object_detector.stop()
        
def estimate_objects_task(cam_data, stop_event, img_pipe, object_detections_pipe, estimated_objects_pipe):
    object_pose_estimator = ope.get_pose_estimator("ycbv",
                                                        cam_data,
                                                        use_tracking = True,
                                                        fuse_detections=False)
    while True:
        t = time.time()
        if stop_event.is_set():
            break
        if img_pipe.poll():
            while img_pipe.poll():
                # print('estimate_objects_task: got img')
                my_img = img_pipe.recv()['img']
        else:
            my_img = img_pipe.recv()['img']
        # print('estimate_objects_task: got img')
        # print(my_img.shape)
        if object_detections_pipe.poll():
            while object_detections_pipe.poll():
                # print('estimate_objects_task: got detected objects')
                my_object_detections = object_detections_pipe.recv()['detected_objects']
        else:
            my_object_detections = None
        # print('estimate_objects_task: got objects')
        # print(my_object_detections)
        estimated_objects = object_pose_estimator.estimate(my_img, detections = my_object_detections)
        # print('estimate_objects_task: got estimated objects')
        if estimated_objects is not None:
            # print('estimate_objects_task: got estimated objects')
            estimated_objects_pipe.send({'estimated_objects': estimated_objects})
            # print('estimate_objects_task: sent estimated objects')
            # print(estimated_objects)
        # print('estimate_objects_task: updated estimated objects')
        print(f'estimate_objects_task: {(time.time()-t)*1000:.2f} ms')
        
    object_pose_estimator.stop()
        

def scene_analysis_task(cam_data, stop_event, detect_event, img_pipe, out_hands, out_object_estimation):
    plotter = pl.NBPlot()
    scene = sc.LiveScene(cam_data, name='Full tracking', plotter=plotter)
    while True:
        # HANDS
        t = time.time()
        detected_hands = None
        while out_hands.poll():
            detected_hands = out_hands.recv()['hands']
            # print(f'got hands')
            # print(detected_hands)
        if detected_hands is not None:
            scene.update_hands(detected_hands)
            print(f'scene update hands: {(time.time()-t)*1000:.2f} ms')
                # print(f'updated hands')
        
        # OBJECTS
        t = time.time()
        estimated_objects = None
        # print('waiting for estimated objects')
        # print(out_object_estimation.poll())
        while out_object_estimation.poll():
            estimated_objects = out_object_estimation.recv()['estimated_objects']
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
        while img_pipe.poll():
            img = img_pipe.recv()['img']
        if img is not None:
            scene.render(img)
            print(f'scene render: {(time.time()-t)*1000:.2f} ms')
            cv2.imshow('render_img', img)
            print(f'updated img : {(time.time()-t)*1000:.2f} ms')
        if k == 27:
            print('end')
            break
    stop_event.set()
        
class GraspingDetector:
    def __init__(self, ) -> None:
        self.dataset = "ycbv"
        
    
    def run(self):
        tracemalloc.start()
        multiprocessing.set_start_method('spawn', force=True)
        dataset = "ycbv"
        rgbd_cam = rgbd.RgbdCamera()
        cam_data = rgbd_cam.get_device_data()
        hands = ['left']
        
        
        plotter = pl.NBPlot()
        scene = sc.LiveScene(cam_data, name='Full tracking', plotter=plotter)
        
        stop_event = multiprocessing.Event()
        detect_event = multiprocessing.Event()
        
        out_rgbd_frame_hands, in_rgbd_frame_hands = multiprocessing.Pipe(duplex=False)
        out_rgb_frame_object_detection, in_rgb_frame_object_detection = multiprocessing.Pipe(duplex=False)
        out_rgb_frame_object_estimation, in_rgb_frame_object_estimation = multiprocessing.Pipe(duplex=False)
        out_rgb_frame_for_scene, in_rgb_frame_for_scene = multiprocessing.Pipe(duplex=False)
        
        out_hands, in_hands = multiprocessing.Pipe(duplex=False)
        out_object_detection, in_object_detection = multiprocessing.Pipe(duplex=False)
        out_object_estimation, in_object_estimation = multiprocessing.Pipe(duplex=False)
        
        
        process_hands_detection = multiprocessing.Process(target=detect_hands_task, 
                                                          args=(cam_data, hands, stop_event, out_rgbd_frame_hands, in_hands,))
        
        process_object_detection = multiprocessing.Process(target=detect_objects_task, 
                                                           args=(cam_data, stop_event, detect_event, out_rgb_frame_object_detection, in_object_detection,))
        
        process_object_estimation = multiprocessing.Process(target=estimate_objects_task, 
                                                            args=(cam_data, stop_event, out_rgb_frame_object_estimation, out_object_detection, in_object_estimation,))
        
        process_scene_analysis = multiprocessing.Process(target=scene_analysis_task, 
                                                        args=(cam_data, stop_event, detect_event, out_rgb_frame_for_scene, out_hands, out_object_estimation,))
        
        process_hands_detection.start()
        process_object_detection.start()
        process_object_estimation.start()
        process_scene_analysis.start()
        rgbd_cam.start()
        
        obj_path = './YCBV_test_pictures/javel.png'
        obj_path2 = './YCBV_test_pictures/mustard_front.png'
        # obj_path = './YCBV_test_pictures/YCBV.png'
        obj_img = cv2.imread(obj_path)
        obj_img = cv2.resize(obj_img, (int(obj_img.shape[1]/2), int(obj_img.shape[0]/2)))
        obj_img2 = cv2.imread(obj_path2)
        obj_img2 = cv2.resize(obj_img2, (int(obj_img2.shape[1]/2), int(obj_img2.shape[0]/2)))
        
        detect_event.set()
        
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
            rgbd_frame = {'img': img_for_hands, 'depth_map': depth_map}
            in_rgbd_frame_hands.send(rgbd_frame)
            # print(f'updated img for hands')
            
            # # OBJECTS
            img[0:obj_img.shape[0], 0:obj_img.shape[1]] = obj_img
            img[0:obj_img2.shape[0], img.shape[1]-obj_img2.shape[1]:] = obj_img2
            
            if detect_event.is_set():
                img_for_objects = img.copy()
                img_for_objects = cv2.cvtColor(img_for_objects, cv2.COLOR_RGB2BGR)
                img_for_objects.flags.writeable = False
                # print(f'sending img for objects detection') 
                if not out_rgb_frame_object_detection.poll():
                    in_rgb_frame_object_detection.send({'img': img_for_objects})
                    # print(f'updated img for objects detection')
        
            img_for_objects = img.copy()
            img_for_objects = cv2.cvtColor(img_for_objects, cv2.COLOR_RGB2BGR)
            img_for_objects.flags.writeable = False
            if not out_rgb_frame_object_estimation.poll():
                in_rgb_frame_object_estimation.send({'img': img_for_objects})
                # print(f'updated img for objects estimation')
            
            # SCENE
            img_for_scene = img.copy()
            t = time.time()
            if not out_rgb_frame_for_scene.poll():
                print(f'polling time : {(time.time()-t)*1000:.2f} ms')
                t= time.time()
                in_rgb_frame_for_scene.send({'img': img_for_scene})
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
        process_hands_detection.terminate()
        rgbd_cam.stop()
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

