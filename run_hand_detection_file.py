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
from experimental_tools.video_utils import concatenate_videos_horizontally

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
        
        cam_data_path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1/PJW0779/cam_1944301011EA1F1300_1280_720_data.npz'
        
        trial_path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1_processing/Pre_processing/NFR4534/trial_9_combi_mustard_left_palmar_simulated'
        file_name = 'trial_9_combi_mustard_left_palmar_simulated_cam_1944301011EA1F1300'
        
        # trial_path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1_processing/Pre_processing/GKX5203/trial_0_combi_cheez\'it_left_pinch_simulated'
        # # trial_path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1//GKX5203/trial_0_combi_cheez\'it_left_pinch_simulated'
        # file_name = 'trial_0_combi_cheez\'it_left_pinch_simulated_cam_1944301011EA1F1300'
        
        # trial_path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1/ODQ2735/trial_0_combi_mustard_left_pinch_simulated'
        # file_name = 'trial_0_combi_mustard_left_pinch_simulated_cam_1944301011EA1F1300'
        
        # trial_path = '/home/emoullet/Documents/i-GRIP/DATA/Session_1/NQH9598/trial_0_combi_tomato_right_palmar_simulated'
        # file_name = 'trial_0_combi_tomato_right_palmar_simulated_cam_1944301011EA1F1300'
        
        rgbd_cam = rgbd.RgbdReader(cam_data_path)
        cam_data = rgbd_cam.get_device_data()
        print(f'cam_data: {cam_data}')
        cam_data_2 = {}
        for key in cam_data:
            cam_data_2[key] = cam_data[key]
        if cam_data_2['resolution'][0] > cam_data_2['resolution'][1]:
            cam_data_2['resolution'] = (cam_data_2['resolution'][1], cam_data_2['resolution'][0])
            print('flipped resolution')
        print(f'cam_data_2: {cam_data_2}')
        use_gpu = True
        fps=5
        scene = sc.LiveScene(cam_data, name='Full tracking', plotter=None, )
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        for use_gpu in (False, False):
        # for use_gpu in (True, False):
            hand_detector = hd.Hands3DDetector(cam_data, hands = self.hands, running_mode =
                                                hd.Hands3DDetector.VIDEO_FILE_MODE, use_gpu=use_gpu)
            if use_gpu:
                path_vid = trial_path + '/full_tracking_gpu.avi'
            else:
                path_vid = trial_path + '/full_tracking_cpu.avi'
            recorder = cv2.VideoWriter(path_vid, fourcc, fps, cam_data['resolution'])
            hand_detector.reset()
            # rgbd_cam.load(trial_path, file_name)
            rgbd_cam.load(trial_path, file_name, suffix='movement')
            for timestamp in rgbd_cam.get_timestamps():
                t = time.time()
                success, img, depth_map = rgbd_cam.next_frame()
                print(f'imgshape: {img.shape}')
                print(f'depth_mapshape: {depth_map.shape}')
                if img.shape[0] > img.shape[1]:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                print(f'depthmap shape: {depth_map.shape}')
                print(f'frame collection time : {(time.time()-t)*1000:.2f} ms')
                if not success:
                    continue
                
                # show depth map
                
                depthFrameColor = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
                depthFrameColor = cv2.resize(depthFrameColor, (depthFrameColor.shape[1]//2, depthFrameColor.shape[0]//2))
                cv2.imshow(f'depth ', depthFrameColor)
                
                # HANDS
                img_for_hands = img.copy()
                img_for_hands = cv2.cvtColor(img_for_hands, cv2.COLOR_RGB2BGR)
                img_for_hands.flags.writeable = False
                rgbd_frame = (img_for_hands, depth_map)
                
                
                # print('detect_hands_task: got img')
                # print(my_img)
                t = time.time()
                estimated_hands = hand_detector.get_hands(*rgbd_frame, timestamp)
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
                    recorder.write(img)
                if k == 27:
                    print('end')
                    break
                print(f'scene analysis task: {(time.time()-t_s)*1000:.2f} ms')
        scene.stop()
        recorder.release()
        concatenate_videos_horizontally(trial_path + '/full_tracking_gpu.avi', trial_path + '/full_tracking_cpu.avi', trial_path + '/full_tracking_comparison.avi', label1='GPU', label2='CPU')
        
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

