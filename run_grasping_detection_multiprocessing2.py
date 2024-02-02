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
        self.dataset = "ycbv"
        self.hand_detector = hd.HybridOAKMediapipeDetector()
        self.cam_data = self.hand_detector.get_device_data()
        self.object_detections = None
        self.is_hands = False
        self.img_for_objects = None
        self.wait_time = 1
        self.lock_hands = multiprocessing.Lock()
        self.lock_objects = multiprocessing.Lock()
        self.lock_estimations = multiprocessing.Lock()

    def detect_hands_task(self, stop_event, img_queue, hand_detections_queue):
        while True:
            if stop_event.is_set():
                break
            my_img = img_queue.get()
            print('detect_hands_task: got img')
            detected_hands = self.hand_detector.get_hands(my_img)
            print('detect_hands_task: got hands')
            with self.lock_hands:
                if hand_detections_queue.full():
                    print('detect_hands_task: queue is full')
                    hand_detections_queue.get()
                    print('detect_hands_task: deleted last hand')
                hand_detections_queue.put(detected_hands)
                print('detect_hands_task: put hands')

    def detect_objects_task(self, detect_event, stop_event, img_queue, object_detections_queue):
        self.object_detector = o2d.get_object_detector(self.dataset, self.cam_data)
        while True:
            if stop_event.is_set():
                break
            detect_flag = detect_event.wait(1)
            print(f'detect_objects_task: detect_event = {detect_flag}')
            if detect_flag:
                img = img_queue.get()
                print('detect_objects_task: got img')
                object_detections = self.object_detector.detect(img)
                print('detect_objects_task: got detections')
                with self.lock_objects:
                    if object_detections_queue.full():
                        print('detect_objects_task: queue is full')
                        object_detections_queue.get()
                        print('detect_objects_task: deleted last detections')
                    object_detections_queue.put(object_detections)
                    print('detect_objects_task: put detections')

    def estimate_objects_task(self, stop_event, img_queue, object_detections_queue, object_pose_estimations_queue):
        self.object_pose_estimator = ope.get_pose_estimator(self.dataset, self.cam_data, use_tracking=True, fuse_detections=False)
        while True:
            if stop_event.is_set():
                break
            img = img_queue.get()
            print('estimate_objects_task: got img')
            with self.lock_objects:
                object_detections = object_detections_queue.get()
                print('estimate_objects_task: got detections')
            object_pose_estimations = self.object_pose_estimator.estimate(img, detections=object_detections)
            with self.lock_estimations:
                if object_pose_estimations_queue.full():
                    print('estimate_objects_task: queue is full')
                    object_pose_estimations_queue.get()
                    print('estimate_objects_task: deleted last estimations')
                object_pose_estimations_queue.put(object_pose_estimations)
                print('estimate_objects_task: put estimations')

    def scene_analysis_task(self, stop_event, img_queue, hand_detections_queue, object_pose_estimations_queue):
        self.plotter = pl.NBPlot()
        self.scene = sc.LiveScene(self.cam_data, name='Full tracking', plotter=self.plotter)
        while True:
            if stop_event.is_set():
                break
            img = img_queue.get()
            print('scene_analysis_task: got img')
            with self.lock_hands:
                hand_detections = hand_detections_queue.get()
                print('scene_analysis_task: got hands')
            with self.lock_estimations:
                object_pose_estimations = object_pose_estimations_queue.get()
                print('scene_analysis_task: got estimations')
            with self.lock_hands, self.lock_estimations:
                print('scene_analysis_task: updating scene')
                self.scene.update_hands(hand_detections)
                self.scene.update_objects(object_pose_estimations)
                print('scene_analysis_task: rendering scene')
                self.scene.render(img)
                cv2.imshow('render_img', img)

    def run(self):
        print(self.__dict__)
        print('start')
        self.hand_detector.start()

        # Initialiser les queues pour la communication entre les processus
        input_queue_img = multiprocessing.Queue(maxsize=1)
        input_queue_img_hands = multiprocessing.Queue(maxsize=1)
        input_queue_img_object_detection = multiprocessing.Queue(maxsize=1)
        input_queue_img_object_estimation = multiprocessing.Queue(maxsize=1)
        input_queue_hands = multiprocessing.Queue(maxsize=1)
        input_queue_detected_objects = multiprocessing.Queue(maxsize=1)
        input_queue_estimated_objects = multiprocessing.Queue(maxsize=1)

        stop_event = multiprocessing.Event()
        detect_event = multiprocessing.Event()
        detect_event.set()

        process_hands_detection = multiprocessing.Process(target=self.detect_hands_task, args=(stop_event, input_queue_img_hands, input_queue_hands))
        process_object_detection = multiprocessing.Process(target=self.detect_objects_task, args=(detect_event, stop_event, input_queue_img_object_detection, input_queue_detected_objects))
        process_object_estimation = multiprocessing.Process(target=self.estimate_objects_task, args=(stop_event, input_queue_img_object_estimation, input_queue_detected_objects, input_queue_estimated_objects))
        process_scene_analysis = multiprocessing.Process(target=self.scene_analysis_task, args=(stop_event, input_queue_img, input_queue_hands, input_queue_estimated_objects))

        process_hands_detection.start()
        process_object_detection.start()
        process_object_estimation.start()
        process_scene_analysis.start()

        obj_path = './YCBV_test_pictures/javel.png'
        obj_img = cv2.imread(obj_path)
        obj_img = cv2.resize(obj_img, (int(obj_img.shape[0] / 2), int(obj_img.shape[1] / 2)))
        i = 0
        while self.hand_detector.isOn():
            k = cv2.waitKey(1)
            success, img = self.hand_detector.next_frame()
            print(f'next_frame : {i}')
            i += 1
            if not success:
                img = None
                print('no image')
                continue
            else:
                img_for_hands = img.copy()
                img_for_hands = cv2.cvtColor(img_for_hands, cv2.COLOR_RGB2BGR)
                img_for_hands.flags.writeable = False
                with self.lock_hands:
                    print(f'putting img in hands queue: {input_queue_img_hands.qsize()}')
                    input_queue_img_hands.put(img_for_hands)
                    print(f'put img in hands queue: {input_queue_img_hands.qsize()}')

                img_for_objects = img.copy()
                img_for_objects[0:obj_img.shape[0], 0:obj_img.shape[1]] = obj_img
                img_for_objects = cv2.cvtColor(img_for_objects, cv2.COLOR_RGB2BGR)
                img_for_objects.flags.writeable = False

                with self.lock_objects:
                    print(f'putting img in objects detect queue: {input_queue_img_object_detection.qsize()}')
                    if input_queue_img_object_detection.full():
                        print('object detection queue is full')
                        input_queue_img_object_detection.get()
                        print('deleted last img in object detection queue')
                    input_queue_img_object_detection.put(img_for_objects)
                    print(f'put img in objects detect queue: {input_queue_img_object_detection.qsize()}')

                    print(f'putting img in objects estimation queue: {input_queue_img_object_estimation.qsize()}')
                    if input_queue_img_object_estimation.full():
                        print('object estimation queue is full')
                        input_queue_img_object_estimation.get()
                        print('deleted last img in object estimation queue')
                    input_queue_img_object_estimation.put(img_for_objects)
                    print(f'put img in objects estimation queue: {input_queue_img_object_estimation.qsize()}')

                with self.lock_hands, self.lock_estimations:
                    print(f'putting img in scene queue: {input_queue_img.qsize()}')
                    if input_queue_img.full():
                        print('scene queue is full')
                        # input_queue_img.get()
                        print('deleted last img in scene queue')
                    input_queue_img.put(img)
                    print(f'put img in scene queue: {input_queue_img.qsize()}')

            if k == 32:
                print('DETEEEEEEEEEEEEEEEEEECT')
                detect_event.set()

            if k == 27:
                print('end')
                self.stop()
                stop_event.set()
                break

        # Attendre la fin de chaque processus avant de terminer le programme
        process_hands_detection.join()
        process_object_detection.join()
        process_object_estimation.join()
        process_scene_analysis.join()

        cv2.destroyAllWindows()
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
