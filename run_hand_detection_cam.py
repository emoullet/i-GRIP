#!/usr/bin/env python3
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import argparse
import os
import yaml

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

_DEFAULT_CAM_SETTINGS_PATH = '/home/emoullet/Mediapipe2/default_cam_settings.yaml'
# Create a hand landmarker instance with the live stream mode:
    

class tset:
    def __init__(self,args) -> None:
        # For webcam input:
        if args.cam_settings == 'void' :
            cam_settings = _DEFAULT_CAM_SETTINGS_PATH
        elif os.path.isfile(args.cam_settings):
            cam_settings = args.cam_settings
        else:
            print('--cam_settings must be the path to a file')
            raise ValueError(args.cam_settings)
        with open(cam_settings) as f:
            self.cam_settings = yaml.safe_load(f)
        self.webcam_index = self.cam_settings['camera0'] # 0 for integrated or 4 for usb cam
        self.detection_result=None
        pass
    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # print('hand landmarker result: {}'.format(result))
        self.detection_result = result
        # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        if detection_result is not None:
            hand_landmarks_list = detection_result.hand_landmarks
            handedness_list = detection_result.handedness
            annotated_image = np.copy(rgb_image)

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                handedness = handedness_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

                # Get the top left corner of the detected hand's bounding box.
                height, width, _ = annotated_image.shape
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height) - MARGIN

                # Draw handedness (left or right hand) on the image.
                cv2.putText(annotated_image, f"{handedness[0].category_name}",
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            return annotated_image
        else:
            return rgb_image

    def run(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='/home/emoullet/Mediapipe2/hand_landmarker.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result,
            num_hands=2)
        
        cap = cv2.VideoCapture(self.webcam_index,cv2.CAP_V4L2)
        # cap.set(3, self.cam_settings['frame_width'])
        # cap.set(4, self.cam_settings['frame_height'])
        expe_img = cv2.imread('/home/emoullet/Documents/i-GRIP/DATA/Session_1_processing/Pre_processing/RKS6625/vids_frames/ExperimentAnaliser_24.png')
        
        with HandLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        # ...
            while cap.isOpened():
                # success, image = cap.read()
                success = True
                image = expe_img
                frame_timestamp_ms = round(time.time()*1000)
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                # cv2.cvtColor(image, cv2.COLOR_RGB2BGR, image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.flip(image,1))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                landmarker.detect_async(mp_image, frame_timestamp_ms)
                annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), self.detection_result)
                print(annotated_image.shape)
                cv2.imshow('tseet', annotated_image)
                # cv2.imshow('tseet',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
                if cv2.waitKey(5) & 0xFF == 27:
                  break

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--cam_settings', default='void', type=str)
    args = parser.parse_args()
    ts = tset(args)
    ts.run()