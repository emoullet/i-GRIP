import mediapipe as mp
import cv2
import numpy as np
import math
import time
from i_grip.config import _MEDIAPIPE_MODEL_PATH
# import tensorflow as tf
# print('TENSORFLOW GPU AVAILABLE:')
# print(tf.config.list_physical_devices('GPU'))

class Hands3DDetector:
    
    LIVE_STREAM_MODE = 'LIVE_STREAM'
    VIDEO_FILE_MODE = 'VIDEO'
    _HANDS_MODE = ['left', 'right']
    
    def __init__(self, cam_data, hands = _HANDS_MODE,running_mode = LIVE_STREAM_MODE,  mediapipe_model_path=_MEDIAPIPE_MODEL_PATH, use_gpu=True):
        self.cam_data = cam_data
        
        for hand in hands:
            if hand not in self._HANDS_MODE:
                raise ValueError(f'hand must be one of {self._HANDS_MODE}')
        self.hands_to_detect = hands
        self.num_hands = len(hands)
        # print("GPU available:", mp.python._pywrap_util.is_gpu_available())

        
        if running_mode not in [self.LIVE_STREAM_MODE, self.VIDEO_FILE_MODE]:
            raise ValueError(f'running_mode must be one of {self.LIVE_STREAM_MODE} or {self.VIDEO_FILE_MODE}')
        
        if use_gpu:            
            base_options=mp.tasks.BaseOptions(model_asset_path=mediapipe_model_path,
                                              delegate = mp.tasks.BaseOptions.Delegate.GPU)
        else:
            base_options=mp.tasks.BaseOptions(model_asset_path=mediapipe_model_path)
            
        if running_mode == self.LIVE_STREAM_MODE:
            self.get_hands = self.get_hands_live_stream
            self.landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=self.num_hands,
                min_hand_presence_confidence=0.5,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self.extract_hands,
            )
        elif running_mode == self.VIDEO_FILE_MODE:
            self.get_hands = self.get_hands_video
            self.landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=self.num_hands,
                min_hand_presence_confidence=0.5,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
        self.init_landmarker()
        self.format=mp.ImageFormat.SRGB
        self.stereoInference = StereoInference(self.cam_data)
        
    def init_landmarker(self):
        self.hands_predictions = []
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(self.landmarker_options)
        
    def reset(self):
        self.init_landmarker()
        self.new_frame = False
        
    def extract_hands(self, detection_result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if detection_result is not None:
            hands_preds = []
            hand_landmarks_list = detection_result.hand_landmarks
            hand_world_landmarks_list = detection_result.hand_world_landmarks
            handedness_list = detection_result.handedness

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                hand_world_landmarks = hand_world_landmarks_list[idx]
                handedness = handedness_list[idx]
                label = handedness[0].category_name.lower()
                if len(hand_landmarks)>0 and self.depth_map is not None and label in self.hands_to_detect and label in self.hands_to_detect:
                    hand = HandPrediction(handedness, hand_landmarks, hand_world_landmarks, self.depth_map, self.stereoInference)
                    hands_preds.append(hand)
            self.hands_predictions = hands_preds

    def get_hands_video(self, frame, depth_frame, timestamp):
        if frame is not None and depth_frame is not None:
        # if frame is not None and depthFrame is not None:
            # mp_frame = cv2.cvtColor(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            # mp_frame = cv2.cvtColor(cv2.flip(self.frame,1), cv2.COLOR_BGR2RGB) 
            # mp_frame = cv2.flip(frame,1)
            # mp_frame=self.frame
            frame_timestamp_ms = round(timestamp*1000)
            mp_image = mp.Image(image_format=self.format, data=frame)
            print(f'frame_timestamp_ms: {frame_timestamp_ms}')
            landmark_results = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            # landmark_results = self.landmarker.detect(mp_image)
            self.depth_map = depth_frame
            self.extract_hands(landmark_results, mp_image, frame_timestamp_ms)
            self.new_frame = False
        return self.hands_predictions
    
    def get_hands_live_stream(self, frame, depth_frame):
        if frame is not None and depth_frame is not None:
        # if frame is not None and depthFrame is not None:
            # mp_frame = cv2.cvtColor(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            # mp_frame = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)
            # mp_frame = cv2.flip(frame,1)
            # print('frame.shape', frame.shape)
            # print(frame)
            # mp_frame=self.frame
            frame_timestamp_ms = round(time.time()*1000)
            # print('frame_timestamp_ms', frame_timestamp_ms)
            mp_image = mp.Image(image_format=self.format, data=frame)
            self.depth_map = depth_frame
            self.landmarker.detect_async(mp_image, frame_timestamp_ms)
            self.new_frame = False
        return self.hands_predictions


class HandPrediction:
    def __init__(self, handedness, landmarks, world_landmarks, depth_map, stereo_inference) -> None:
        self.handedness = handedness
        self.normalized_landmarks = np.array([[l.x,l.y,l.z] for l in landmarks])
        # self.landmarks = np.array([[max(min(1-l.x,1.),0.)*img_res[0], max(min(l.y,1.),0.)*img_res[1], l.z] for l in landmarks])
        # self.normalized_landmarks = landmarks
        # print('landmarks', landmarks)
        # print('world_landmarks', world_landmarks)
        self.world_landmarks = np.array([[l.x,-l.y,l.z] for l in world_landmarks])*1000
        # print('self.world_landmarks', self.world_landmarks)
        # self.world_landmarks = np.array([[l.x*img_res[0], l.y*img_res[1], l.z] for l in world_landmarks])
        self.label = handedness[0].category_name.lower()
        hand_point2D, hand_point3D = self.hand_point()
        self.position, self.roi = stereo_inference.calc_spatials(hand_point2D, depth_map)
        #add self.position to every world_landmarks lines
        hand_center = self.position.copy()
        # hand_center[1] = -hand_center[1]
        self.world_landmarks = self.world_landmarks + hand_center - hand_point3D
        # self.position = self.position/1000
        
    def hand_point(self):
        # hand_point2D = self.normalized_landmarks[0,:] # wrist
        # hand_point3D = self.world_landmarks[0,:] # wrist
        hand_point2D = (self.normalized_landmarks[0,:]+self.normalized_landmarks[5,:]+self.normalized_landmarks[17,:])/3 # wrist, index finger and pinky baricenter
        hand_point3D = (self.world_landmarks[0,:]+self.world_landmarks[5,:]+self.world_landmarks[17,:])/3 # wrist, index finger and pinky baricenter
        return hand_point2D, hand_point3D

    def get_landmarks(self):
        return self.normalized_landmarks
    

class StereoInference:
    def __init__(self, cam_data) -> None:

        self.original_width = cam_data['resolution'][0]
        self.original_height = cam_data['resolution'][1]
        
        self.hfov = cam_data['hfov']
        self.hfov = np.deg2rad(self.hfov)

        self.depth_thres_high = 3000
        self.depth_thres_low = 50
        self.box_size = 10


    def calc_angle(self, offset):
            return math.atan(math.tan(self.hfov / 2.0) * offset / (self.original_width / 2.0))

    def calc_spatials(self, normalized_img_point, depth_map, averaging_method=np.mean):
        if depth_map is None:
            print('No depth map available yet')
            return np.array([0,0,0]), None
        #box_size = max(5, int(np.linalg.norm(wrist-thumb)))/2
        # print('depth_map.shape',depth_map.shape)
        x = normalized_img_point[0]*self.original_width
        y = normalized_img_point[1]*self.original_height
        xmin = max(int(x-self.box_size),0)
        xmax = min(int(x+self.box_size), int(depth_map.shape[1]))
        ymin = max(int(y-self.box_size),0 )
        ymax = min(int(y+self.box_size), int(depth_map.shape[0]))
        
        if xmin > xmax:  # bbox flipped
            xmin, xmax = xmax, xmin
        if ymin > ymax:  # bbox flipped
            ymin, ymax = ymax, ymin

        if xmin == xmax : 
            xmax = xmin +self.box_size
        if ymin == ymax :
            ymax = ymin +self.box_size

        # Calculate the average depth in the ROI.
        depthROI = depth_map[ymin:ymax, xmin:xmax]
        # print('depthROI', depthROI)
        inThreshRange = (self.depth_thres_low < depthROI) & (depthROI < self.depth_thres_high)
        if depthROI[inThreshRange].any():
            averageDepth = averaging_method(depthROI[inThreshRange])
        else:
            averageDepth = 0
        #print(f"Average depth: {averageDepth}")

        mid_w = int(depth_map.shape[1] / 2) # middle of the depth img
        mid_h = int(depth_map.shape[0] / 2) # middle of the depth img
        bb_x_pos = x - mid_w
        bb_y_pos = y - mid_h
        # print('wrist',wrist)
        angle_x = self.calc_angle(bb_x_pos)
        angle_y = self.calc_angle(bb_y_pos)

        z = averageDepth
        x = z * math.tan(angle_x)
        y = -z * math.tan(angle_y)

        # print(f"DEPTH MAP --- X: {x/10:3.0f}cm, Y: {y/10:3.0f} cm, Z: {z/10:3.0f} cm")
        return np.array([x,y,z]), (xmin, ymin, xmax, ymax)
    

