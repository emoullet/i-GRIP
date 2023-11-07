#!/usr/bin/env python3

import argparse
from i_grip import HandDetectors2 as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene as sc
import cv2
import numpy as np

class ExperimentReplayer:
    def __init__(self, device_id, device_data, name = None, display_replay = True, resolution=(1280,720), fps=30.0) -> None:
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        
        dataset = "ycbv"        
        self.display_replay = display_replay
        
        device_data = np.load('/home/emoullet/Documents/i-GRIP/DATA/Session_1/cam_19443010910F481300.npz')
        self.object_detector = o2d.get_object_detector(dataset,
                                                       device_data)
        self.object_pose_estimator = ope.get_pose_estimator(dataset,
                                                            device_data,
                                                            use_tracking = True,
                                                            fuse_detections=False)
        if name is None:
            self.name = f'ExperimentReplayer_{dataset}'
        else:
            self.name = name
        self.scene = sc.LiveScene( device_data, name = f'{self.name}_scene')
    
    def get_device_id(self):
        return self.device_id
    
    

    def replay(self, gh):
        print(self.__dict__)
        print('start')
        detect = True
        obj_path = './YCBV_test_pictures/mustard_back.png'
        obj_path = './YCBV_test_pictures/YCBV2.png'
        obj_path = './YCBV_test_pictures/cap2.png'
        obj_img = cv2.imread(obj_path)
        # obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)
        while True:
            img = obj_img
            if False:
                pass
            else:
                render_img = img.copy()
                to_process_img = img.copy()
                cv2.cvtColor(to_process_img, cv2.COLOR_RGB2BGR, to_process_img)
                
                    #replace pixels from self.img with obj_img
                    
                if detect:
                    self.object_detections = self.object_detector.detect(to_process_img)
                    if self.object_detections is not None:
                        detect = False
                        print(f'detect {self.object_detections.bboxes}')
                    print('detect')
                else:
                    self.object_detections = None

                # Object pose estimation
                self.objects_pose = self.object_pose_estimator.estimate(to_process_img, detections = self.object_detections)
                print(f'pose {self.objects_pose}')
                self.scene.update_objects(self.objects_pose)
            print('lilililiiiiiii')
            k = cv2.waitKey(20)
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
            self.scene.render(render_img)
            cv2.imshow('image', render_img)
            if k==27:
                print('end')
                self.stop()
                break
    
    def run(self, replay, name = None):
        
        
        if name is not None:
            cv_window_name = f'{self.name} : Replaying {name}'
        else:
            cv_window_name = f'{self.name} : Replaying'
        detect = True
        for i in range(50):
            obj_path = './YCBV_test_pictures/cap2.png'
            img = cv2.imread(obj_path)
            render_img = img.copy()
            to_process_img = img.copy()
            cv2.cvtColor(to_process_img, cv2.COLOR_RGB2BGR, to_process_img)
            # to_process_img.flags.writeable = False

            # Hand detection
            # hands = self.hand_detector.get_hands(to_process_img)
            #self.scene.update_hands(hands, timestamp)

            # Object detection
            if detect:
                self.object_detections = self.object_detector.detect(to_process_img)
                if self.object_detections is not None:
                    detect = False
                print('detect')
            else:
                self.object_detections = None

            # Object pose estimation
            self.objects_pose = self.object_pose_estimator.estimate(to_process_img, detections = self.object_detections)
            self.scene.update_objects(self.objects_pose)
            #img.flags.writeable = False
            # img.flags.writeable = True
            # cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            if self.display_replay:
                self.scene.render(render_img)
                cv2.imshow(cv_window_name, render_img)
            print('lalalalalalalaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('end')
                self.stop()
                break

        hands_data = self.scene.get_hands_data()
        objects_data = self.scene.get_objects_data()
        
        return hands_data, objects_data
        
    def stop(self):
        print("Stopping experiment replayer...")
        print("Stopping hand detector...")
        print("Stopped hand detector...")
        print("Stopping object estimator...")
        self.object_pose_estimator.stop()
        print("Stopped object estimator...")
        print("Stopping scene...")
        self.scene.stop()
        print("Stopped scene...")
        
        cv2.destroyAllWindows()
        

        

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
    i_grip = ExperimentReplayer(**args)
    i_grip.run()