#!/usr/bin/env python3

import argparse
import cv2
import time 
import os
from i_grip import RgbdCameras as rgbd
from i_grip import Hands3DDetectors as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene_refactored_multi as sc

class ExperimentReplayer:
    def __init__(self, device_id, device_data, name = None, display_replay = True, resolution=(1280,720), fps=30.0) -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.device_id = device_id
        self.resolution = resolution
        print(f'resolution: {self.resolution}')
        # self.resolution = (int(self.resolution[1]), int(self.resolution[0]))
        self.fps = fps
        
        dataset = "ycbv"        
        self.display_replay = display_replay
        
        self.rgbd_cam = rgbd.RgbdCamera(replay=True, cam_params= device_data, resolution=self.resolution, fps=fps)
        cam_data = self.rgbd_cam.get_device_data()
        print(f'cam_data: {cam_data}')
        cam_data_2 = {}
        for key in cam_data:
            cam_data_2[key] = cam_data[key]
        cam_data_2['resolution'] = (self.resolution[1], self.resolution[0])
        hands = ['right', 'left']
        self.hand_detector = hd.Hands3DDetector(cam_data, hands = hands, running_mode =
                                            hd.Hands3DDetector.VIDEO_FILE_MODE)
        self.object_detector = o2d.get_object_detector(dataset,
                                                       cam_data)
        self.object_pose_estimator = ope.get_pose_estimator(dataset,
                                                            cam_data,
                                                            use_tracking = True,
                                                            fuse_detections=False)
        print('Waiting for the camera to start...')
        time.sleep(4)
        print('Camera started')
        if name is None:
            self.name = f'ExperimentReplayer_{dataset}'
        else:
            self.name = name
        self.scene = sc.ReplayScene( cam_data, name = f'{self.name}_scene', dataset = dataset)
        # self.scene = sc.ReplayScene( device_data, name = f'{self.name}_scene')
        self.rgbd_cam.start()
    
    def get_device_id(self):
        return self.device_id
    
    def replay(self, replay, name = None):
        
        self.rgbd_cam.load_replay(replay)
        self.object_pose_estimator.reset()
        self.scene.reset()
        self.hand_detector.reset()
        if name is not None:
            cv_window_name = f'{self.name} : Replaying {name}'
            print(f'Replaying {name}')
        else:
            cv_window_name = f'{self.name} : Replaying'
        self.detect = True
        print(f'all timestamps: {self.rgbd_cam.get_timestamps()}')
        for timestamp in self.rgbd_cam.get_timestamps():
            success, img, depth_map = self.rgbd_cam.next_frame()
            if not success:
                continue
            # img = cv2.resize(img, (self.resolution[1], self.resolution[0]))
            print(f'timestamp: {timestamp}')
            print(f'img shape: {img.shape}')
            # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
            render_img = img.copy()
            to_process_img = img.copy()
            fac = 2
            
            to_process_img = cv2.cvtColor(to_process_img, cv2.COLOR_RGB2BGR)
            # cv2.cvtColor(to_process_img, cv2.COLOR_RGB2BGR, to_process_img)
            
            # smol_to_process_img = cv2.resize(to_process_img, (int(self.resolution[0]/fac), int(self.resolution[1]/fac)))
            
            # Hand detection
            hands = self.hand_detector.get_hands(to_process_img, depth_map, timestamp)
            # smol_hands = self.hand_detector.get_hands(smol_to_process_img)
            # for hand in smol_hands:
            #     hand.label = hand.label + '_smol'
            self.scene.update_hands(hands, timestamp)

            # Object detection
            if self.detect:
                self.object_detections = self.object_detector.detect(to_process_img)
                if self.object_detections is not None:
                    self.detect = False
                print('detect')
            else:
                self.object_detections = None

            # Object pose estimation
            self.objects_pose = self.object_pose_estimator.estimate(to_process_img, detections = self.object_detections)
            
            # check if all objects are detected
            expected_objects = sc.RigidObject.LABEL_EXPE_NAMES
            for label in expected_objects:
                if label not in self.objects_pose:
                    self.detect = True
                    
            # remove unexpected objects
            keys_to_remove = []
            for label in self.objects_pose:
                if label not in expected_objects:
                    keys_to_remove.append(label)
            for key in keys_to_remove:
                del self.objects_pose[key]
                
            self.scene.update_objects(self.objects_pose, timestamp)
            if self.display_replay:
                self.scene.render(render_img)
                print(f'render_img shape: {render_img.shape}')
                cv2.imshow(cv_window_name, render_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('end')
                self.stop()
                break
        self.scene.pause_scene_display()
        hands_data = self.scene.get_hands_data()
        objects_data = self.scene.get_objects_data()
        
        return hands_data, objects_data
        
    def stop(self):
        print("Stopping experiment replayer...")
        print("Stopping hand detector...")
        self.rgbd_cam.stop()
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