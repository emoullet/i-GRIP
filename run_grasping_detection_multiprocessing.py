#!/usr/bin/env python3

import argparse
import multiprocessing
import os
import cv2

from i_grip import HandDetectors2 as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene2 as sc
from i_grip import Plotters3 as pl
from i_grip.utils import kill_gpu_processes


class GraspingDetector:
    def __init__(self) -> None:
        dataset = "ycbv"
        self.hand_detector = hd.HybridOAKMediapipeDetector()
        cam_data = self.hand_detector.get_device_data()
        plotter = pl.NBPlot()
        self.object_detector = o2d.get_object_detector(dataset, cam_data)
        self.object_pose_estimator = ope.get_pose_estimator(dataset, cam_data, use_tracking=True, fuse_detections=False)
        self.scene = sc.LiveScene(cam_data, name='Full tracking', plotter=plotter)
        self.object_detections = None
        self.is_hands = False
        self.img_for_objects = None

    def estimate_objects_task(self, start_event, estimate_event):
        while self.hand_detector.isOn():
            start_flag = start_event.wait(1)
            if start_flag:
                if estimate_event.wait(1):
                    self.objects_pose = self.object_pose_estimator.estimate(self.img_for_objects, detections=self.object_detections)
                    self.scene.update_objects(self.objects_pose)
                    estimate_event.clear()

    def detect_objects_task(self, start_event, detect_event, estimate_event):
        while self.hand_detector.isOn():
            start_flag = start_event.wait(1)
            if start_flag:
                detect_flag = detect_event.wait(1)
                if detect_flag:
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
        start_event = multiprocessing.Event()
        detect_event = multiprocessing.Event()
        estimate_event = multiprocessing.Event()
        self.t_obj_d = multiprocessing.Process(target=self.detect_objects_task, args=(start_event, detect_event, estimate_event,))
        self.t_obj_e = multiprocessing.Process(target=self.estimate_objects_task, args=(start_event, estimate_event,))
        self.t_obj_d.start()
        self.t_obj_e.start()
        started = True
        obj_path = './YCBV_test_pictures/javel.png'
        obj_img = cv2.imread(obj_path)
        obj_img = cv2.resize(obj_img, (int(obj_img.shape[0] / 2), int(obj_img.shape[1] / 2)))
        while self.hand_detector.isOn():
            k = cv2.waitKey(2)
            success, img = self.hand_detector.next_frame()
            if not success:
                self.img_for_objects = None
                continue
            else:
                img[0:obj_img.shape[0], 0:obj_img.shape[1]] = obj_img
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
                if hands is not None and len(hands) > 0:
                    self.scene.update_hands(hands)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hd', '--hand_detection', choices=['mediapipe', 'depthai', 'hybridOAKMediapipe'],
                        default='hybridOAKMediapipe', help="Hand pose reconstruction solution")
    parser.add_argument('-od', '--object_detection', choices=['cosypose, megapose'],
                        default='cosypose', help="Object pose reconstruction detection")
    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print('start')
    kill_gpu_processes()
    i_grip = GraspingDetector()
    i_grip.run()
