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
        # cam_data_2['resolution'] = (self.resolution[1], self.resolution[0])
        hands = ['right', 'left']
        self.hand_detector = hd.Hands3DDetector(cam_data_2, hands = hands, running_mode =
                                            hd.Hands3DDetector.VIDEO_FILE_MODE,
                                            use_gpu=True)
        self.object_detector = o2d.get_object_detector(dataset,
                                                       cam_data_2)
        self.object_pose_estimator = ope.get_pose_estimator(dataset,
                                                            cam_data_2,
                                                            use_tracking = True,
                                                            fuse_detections=False)
        print('Waiting for the camera to start...')
        time.sleep(4)
        print('Camera started')
        if name is None:
            self.name = f'ExperimentReplayer_{dataset}'
        else:
            self.name = name
        self.scene = sc.ReplayScene( cam_data_2, name = f'{self.name}_scene', dataset = dataset)
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
        failed_detections_count = 0
        split_image = False
        print(f'all timestamps: {self.rgbd_cam.get_timestamps()}')
        for timestamp in self.rgbd_cam.get_timestamps():
            success, img, depth_map = self.rgbd_cam.next_frame()
            width, height = img.shape[1], img.shape[0]
            if height >= width:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
            
            
            # show depth map
            
            depthFrameColor = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            depthFrameColor = cv2.resize(depthFrameColor, (depthFrameColor.shape[1]//2, depthFrameColor.shape[0]//2))
            cv2.imshow(f'depth ', depthFrameColor)
            
            # Hand detection
            hands = self.hand_detector.get_hands(to_process_img, depth_map, timestamp)
            # smol_hands = self.hand_detector.get_hands(smol_to_process_img)
            # for hand in smol_hands:
            #     hand.label = hand.label + '_smol'
            self.scene.update_hands(hands, timestamp)

            # Object detection
            if self.detect:
                if not split_image:
                    self.object_detections = self.object_detector.detect(to_process_img)
                else:
                    half = int(to_process_img.shape[1]/2)
                    print(f'to_process_img shape: {to_process_img.shape}')
                    img1 = to_process_img[:, :int(to_process_img.shape[1]/2)]
                    img2 = to_process_img[:, int(to_process_img.shape[1]/2):]    
                    print(f'img1 shape: {img1.shape}')
                    print(f'img2 shape: {img2.shape}')
                    object_detections1 = self.object_detector.detect(img1)
                    object_detections2 = self.object_detector.detect(img2)
                    print(f'object_detections1: {object_detections1}')
                    print(f'object_detections2: {object_detections2}')
                    bbox1 = object_detections1.bboxes.cpu()
                    bbox2 = object_detections2.bboxes.cpu()
                    print(f'bbox1: {bbox1}')
                    print(f'bbox2: {bbox2}')
                    # add half to x coordinate of bbox2
                    bbox2[:, 0] += half
                    bbox2[:, 2] += half
                    print(f'bbox2: {bbox2}')
                    united_detection = object_detections1
                    infos_u = united_detection.infos
                    bbox_u = united_detection.bboxes.cpu()
                    infos_2 = object_detections2.infos
                    i=0
                    for row in infos_u.iterrows():
                        if row[1]['label'] not in infos_2['label'].values:
                            infos_u.loc[len(infos_u)] = row[1]
                        else:
                            bbox_u[i, 0] = min(bbox_u[i, 0], bbox2[i, 0])
                            bbox_u[i, 2] = min(bbox_u[i, 2], bbox2[i, 2])
                            bbox_u[i, 1] = max(bbox_u[i, 1], bbox2[i, 1])
                            bbox_u[i, 3] = max(bbox_u[i, 3], bbox2[i, 3])
                    united_detection.bboxes = bbox_u.cuda().float()
                    self.object_detections = united_detection
                    
                            
                    
                
                if self.object_detections is not None:
                    self.detect = False
                    failed_detections_count += 1
                if failed_detections_count > 3:
                    split_image = True
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