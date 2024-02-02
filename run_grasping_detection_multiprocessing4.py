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
        
        self.lock_img_for_hands = multiprocessing.Lock()
        self.lock_img_for_object_detections = multiprocessing.Lock()
        self.lock_img_for_object_estimations = multiprocessing.Lock()
        self.lock_object_detection_for_estimations = multiprocessing.Lock()
        self.lock_img_for_scene_analysis = multiprocessing.Lock()
        self.lock_object_estimation_for_scene_analysis = multiprocessing.Lock()
        self.lock_hands_detection_for_scene_analysis = multiprocessing.Lock()
        self.lock_scene = multiprocessing.Lock()
        
        self.input_img_hands = None
        self.input_img_object_detection = None
        self.input_img_object_estimation = None
        self.input_hands = None
        self.input_detected_objects = None
        self.input_estimated_objects = None

    def detect_hands_task(self, stop_event, img_queue, detected_hands_queue):
        while True:
            if stop_event.is_set():
                break
            my_img = img_queue.get()
            print('detect_hands_task: got img')
            print(my_img.shape)
            detected_hands = self.hand_detector.get_hands(my_img)
            if detected_hands is not None:
                print('detect_hands_task: got hands')
                print(detected_hands)
                self.scene.update_hands(detected_hands)
            if not img_queue.empty():
                img_queue.get()
            # print('detect_hands_task: updated hands')

    def detect_objects_task(self, detect_event, stop_event, img_queue, detected_objects_queue):
        self.object_detector = o2d.get_object_detector(self.dataset, self.cam_data)
        while True:
            if stop_event.is_set():
                break
            detect_flag = detect_event.wait(1)
            print(f'detect_objects_task: detect_event = {detect_flag}')
            
            if not detect_flag:
                continue
            if not img_queue.empty():
                with self.lock_img_for_object_detections:
                    img = img_queue.get()
                    print('detect_objects_task: got img')
                
                object_detections = self.object_detector.detect(img)
                if object_detections is not None:
                    print('detect_objects_task: got detections')
                    detected_objects_queue.put(object_detections)
                    print('detect_objects_task: updated detections')
                    detect_event.clear()

    def estimate_objects_task(self, stop_event, img_queue, detected_objects_queue):
        self.object_pose_estimator = ope.get_pose_estimator(self.dataset, self.cam_data, use_tracking=True, fuse_detections=False)
        while True:
            if stop_event.is_set():
                break
            
            if not img_queue.empty():
                with self.lock_img_for_object_estimations:
                    img = img_queue.get()
                    print('estimate_objects_task: got img')
            
            if not detected_objects_queue.empty():
                with self.lock_object_detection_for_estimations:
                    object_detections = detected_objects_queue.get()
                    print('estimate_objects_task: got detections')
                    
            object_pose_estimations = self.object_pose_estimator.estimate(img, detections=object_detections)
            if object_pose_estimations is not None:
                self.scene.update_objects(object_pose_estimations)
            

    def scene_analysis_task(self, stop_event, detect_event, img_queue):
        j =0
        while True:
            j+=1
            if stop_event.is_set():
                break
            
            img = img_queue.get()          
            k = cv2.waitKey(1)          
            self.scene.render(img)
            cv2.imshow('render_img', img)
            if k == 32:
                print('DETEEEEEEEEEEEEEEEEEECT')
                detect_event.set()

            if k == 27:
                print('end')
                self.stop()
                stop_event.set()
                break
            
            # if j >10:
            #     break
            

    def run(self):
        print(self.__dict__)
        print('start')
        self.hand_detector.start()
        self.plotter = pl.NBPlot()
        self.scene = sc.LiveScene(self.cam_data, name='Full tracking', plotter=self.plotter)

        stop_event = multiprocessing.Event()
        detect_event = multiprocessing.Event()
        detect_event.set()
        
        
        input_queue_img = multiprocessing.Queue(maxsize=1)
        input_queue_img_hands = multiprocessing.Queue(maxsize=1)
        input_queue_img_object_detection = multiprocessing.Queue(maxsize=1)
        input_queue_img_object_estimation = multiprocessing.Queue(maxsize=1)
        input_queue_hands = multiprocessing.Queue(maxsize=1)
        input_queue_detected_objects = multiprocessing.Queue(maxsize=1)
        input_queue_estimated_objects = multiprocessing.Queue(maxsize=1)

        process_hands_detection = multiprocessing.Process(target=self.detect_hands_task, args=(stop_event, input_queue_img_hands, input_queue_hands,))
        process_object_detection = multiprocessing.Process(target=self.detect_objects_task, args=(detect_event, stop_event, input_queue_img_object_detection, input_queue_detected_objects,))
        process_object_estimation = multiprocessing.Process(target=self.estimate_objects_task, args=(stop_event, input_queue_img_object_estimation, input_queue_detected_objects,))
        process_scene_analysis = multiprocessing.Process(target=self.scene_analysis_task, args=(stop_event, detect_event, input_queue_img,))

        process_hands_detection.start()
        # process_object_detection.start()
        # process_object_estimation.start()
        process_scene_analysis.start()

        obj_path = './YCBV_test_pictures/javel.png'
        obj_img = cv2.imread(obj_path)
        obj_img = cv2.resize(obj_img, (int(obj_img.shape[0] / 2), int(obj_img.shape[1] / 2)))
        i = 0
        while self.hand_detector.isOn():
            success, img = self.hand_detector.next_frame()
            # print(f'next_frame : {i}')
            i += 1
            if not success:
                img = None
                print('no image')
                continue
            
            
            # HANDS
            if not input_queue_img_hands.full():
                img_for_hands = img.copy()
                img_for_hands = cv2.cvtColor(img_for_hands, cv2.COLOR_RGB2BGR)
                img_for_hands.flags.writeable = False
                input_queue_img_hands.put(img_for_hands)
                print(f'updated img for hands')
            
            # OBJECT DETECTION
            img_for_objects = img.copy()
            img_for_objects = cv2.cvtColor(img_for_objects, cv2.COLOR_RGB2BGR)
            img_for_objects.flags.writeable = False
            
            if not input_queue_img_object_detection.full():
                input_queue_img_object_detection.put(img_for_objects)
            
            if not input_queue_img_object_estimation.full():
                input_queue_img_object_estimation.put(img_for_objects)
            
            # SCENE ANALYSIS
            if not input_queue_img.full():
                input_queue_img.put(img)
            
            # if i>5:
            #     break
                

        process_hands_detection.terminate()
        process_object_detection.terminate()
        process_object_estimation.terminate()
        process_scene_analysis.terminate()
        cv2.destroyAllWindows()
        exit()

    def stop(self):
        self.hand_detector.stop()


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
