#!/usr/bin/env python3

import argparse
from i_grip import HandDetectors2 as hd
from i_grip import Object2DDetectors as o2d
from i_grip import ObjectPoseEstimators as ope
from i_grip import Scene2 as sc
import cv2

class ExperimentReplayer:
    def __init__(self, device_id, device_data, name = None, display_replay = True, resolution=(1280,720), fps=30.0) -> None:
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        
        dataset = "tless"        
        self.display_replay = display_replay
        self.hand_detector = hd.HybridOAKMediapipeDetector(replay=True, cam_params= device_data, resolution=resolution, fps=fps)
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
        self.scene = sc.ReplayScene( device_data, name = f'{self.name}_scene')
        self.hand_detector.start()
    
    def get_device_id(self):
        return self.device_id
    
    def replay(self, replay, name = None):
        
        self.hand_detector.load_replay(replay)
        
        if name is not None:
            cv_window_name = f'{self.name} : Replaying {name}'
        else:
            cv_window_name = f'{self.name} : Replaying'
        detect = True
        
        for timestamp in self.hand_detector.get_timestamps():
            success, img = self.hand_detector.next_frame()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img.flags.writeable = False
            if not success:
                continue

            # Hand detection
            hands = self.hand_detector.get_hands(img)
            self.scene.update_hands(hands, timestamp)

            # Object detection
            if detect:
                self.object_detections = self.object_detector.detect(img)
                if self.object_detections is not None:
                    detect = False
                print('detect')
            else:
                self.object_detections = None

            # Object pose estimation
            self.objects_pose = self.object_pose_estimator.estimate(img, detections = self.object_detections)
            self.scene.update_objects(self.objects_pose, timestamp)
            #img.flags.writeable = False
            img.flags.writeable = True
            if self.display_replay:
                render_img = self.scene.render(img)
                cv2.imshow(cv_window_name, render_img)
            
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
        self.hand_detector.stop()
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