#!/usr/bin/env python3
import mediapipe as mp
import cv2
import numpy as np
import time
import math

import depthai as dai

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class HybridOAKMediapipeDetector():
    
    _MEDIAPIPE_MODEL_PATH = '/home/emoullet/Mediapipe2/hand_landmarker_aout23.task'
    
    _720P = [1280., 720.]
    _1080P = [1920., 1080.]
    _480P = [640., 480.]
    
    def __init__(self, replay = False, 
                 replay_data= None, 
                 cam_params = None, 
                 device_id = None, 
                 fps=30., 
                 resolution = _720P, 
                 detect_hands = True, 
                 mediapipe_model_path = _MEDIAPIPE_MODEL_PATH, 
                 print_rgb_stereo_latency = False, 
                 show_disparity=False) -> None:
        """_summary_

        Args:
            replay (bool, optional): _description_. Defaults to False.
            replay_data (_type_, optional): _description_. Defaults to None.
            cam_params (_type_, optional): _description_. Defaults to None.
            device_id (_type_, optional): _description_. Defaults to None.
            fps (_type_, optional): _description_. Defaults to 30..
            resolution (_type_, optional): _description_. Defaults to _720P.
            detect_hands (bool, optional): _description_. Defaults to True.
            mediapipe_model_path (_type_, optional): _description_. Defaults to _MEDIAPIPE_MODEL_PATH.
            print_rgb_stereo_latency (bool, optional): _description_. Defaults to False.
            show_disparity (bool, optional): _description_. Defaults to False.
        """
        self.type = 'HybridOAKMediapipeDetector'
        self.cam_auto_mode = True
        self.device_id = device_id
        self.replay = replay
        self.fps = fps
        self.resolution = resolution
        self.print_rgb_stereo_latency=print_rgb_stereo_latency
        self.show_disparity=show_disparity
        print(f'fps: {fps}, resolution: {resolution}')
        if self.replay :
            if replay_data is not None:
                self.load_replay(replay_data)
            self.next_frame = self.next_frame_video
            self.get_hands = self.get_hands_video
            self.landmarker_options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=mediapipe_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
                # min_hand_detection_confidence = 0.8,
                # min_hand_presence_confidence = 0.8
                )
            self.isOn = self.isOn_replay
            self.cam_data = cam_params
        else:
            self.cam_data = {}
            self.build_device()
            self.next_frame = self.next_frame_livestream
            self.get_hands = self.get_hands_live_stream
            self.landmarker_options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=mediapipe_model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_hands=2,
                # min_hand_detection_confidence = 0.8,
                # min_hand_presence_confidence = 0.8,
                result_callback=self.extract_hands)

        self.frame = None
        self.new_frame = False
        if detect_hands:
            self.init_landmarker()
            self.format=mp.ImageFormat.SRGB

        self.margin = 10  # pixels
        self.font_size = 1
        self.font_thickness = 1
        self.handedness_text_color = (88, 205, 54) # vibrant green
        self.stereoInference = StereoInference(self.cam_data)
        print(f'Hand detector built: replay={replay}, detect_hands={detect_hands}')

    def get_res(self):
        return self.cam_data['resolution']
    
    def get_device_data(self):
        return self.cam_data
    
    def start(self):
        self.on = True
    
    def isOn(self):
        return self.on
    
    def init_landmarker(self):
        self.detection_result=None
        self.hands_predictions=[]
        self.landmarker = HandLandmarker.create_from_options(self.landmarker_options)
        
    def stop(self):
        self.on = False
        #wait for 50ms
        if not self.replay:
            time.sleep(0.05)
            self.device.close()

    def build_device(self):
        print("Building device...")
        if self.device_id is None:
            self.device = dai.Device()
        else:
            self.device = dai.Device(dai.DeviceInfo(self.device_id))
        self.lensPos = 120
        self.expTime = 8000
        self.sensIso = 400    
        self.wbManual = 4000
        self.rgb_res = dai.ColorCameraProperties.SensorResolution.THE_1080_P
        self.mono_res = dai.MonoCameraProperties.SensorResolution.THE_400_P
        self.cam_data['resolution'] = (int(self.resolution[0]), int(self.resolution[1]))
        print(f'resolution: {self.cam_data["resolution"]}')
        # print(np.array(self.resolution))
        # print(np.array(self.resolution).shape)
        # print(np.array([*self.resolution]))
        # print(np.array([*self.resolution]).shape)
        calibData = self.device.readCalibration()
        self.cam_data['matrix'] = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, self.cam_data['resolution'][0], self.cam_data['resolution'][1]))
        self.cam_data['hfov'] = calibData.getFov(dai.CameraBoardSocket.RGB)
        self.device.startPipeline(self.create_pipeline())

        self.rgbQ = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.depthQ = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        print("Device built.")

    def isOn_replay(self):
        return self.current_frame_index<self.nb_frames
    
    def load_replay(self, replay):
        self.video = cv2.VideoCapture(replay['Video'])
        self.nb_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0
        if not self.video.isOpened():
            print("Error reading video") 
            exit()
        self.timestamps = replay['Timestamps']
        self.depth_maps = replay['Depth_maps']
        self.init_landmarker()
        
    def get_num_frames(self):
        return self.nb_frames
    
    def get_timestamps(self):
        return self.timestamps
    
    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setXLinkChunkSize(0) # decrease latency
        
        # ColorCamera
        print("Creating Color Camera...")
        camRgb = pipeline.createColorCamera()
        camRgb.setResolution(self.rgb_res)
        controlIn = pipeline.create(dai.node.XLinkIn)
        controlIn.setStreamName('control')
        controlIn.out.link(camRgb.inputControl)

        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        # camRgb.setInterleaved(False)
        camRgb.setIspScale(2, 3)
        if self.cam_auto_mode:
            camRgb.initialControl.setAutoExposureEnable()
        else:
            camRgb.initialControl.setManualWhiteBalance(self.wbManual)
            print("Setting manual exposure, time: ", self.expTime, "iso: ", self.sensIso)
            camRgb.initialControl.setManualExposure(self.expTime, self.sensIso)
        camRgb.initialControl.setManualFocus(self.lensPos)
        # cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        camRgb.setFps(self.fps)
        camRgb.setPreviewSize(self.cam_data['resolution'][0], self.cam_data['resolution'][1])
        # camRgb.setVideoSize(self.cam_data['resolution'][0], self.cam_data['resolution'][1])

        
        camLeft = pipeline.create(dai.node.MonoCamera)
        camRight = pipeline.create(dai.node.MonoCamera)
        camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        # cam.setVideoSize(self.cam_data['resolution'])
        for monoCam in (camLeft, camRight):  # Common config
            monoCam.setResolution(self.mono_res)
            monoCam.setFps(self.fps)

        print('ici')

        self.cam_out = pipeline.createXLinkOut()
        self.cam_out.setStreamName("rgb")
        ### uncommenting decreases rgb latency, but since stereo has bigger latency, keeping them commented decreases overall latency between rgb and stereo
        self.cam_out.input.setQueueSize(1)
        self.cam_out.input.setBlocking(False)
        camRgb.isp.link(self.cam_out.input)

        # Create StereoDepth node that will produce the depth map
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.initialConfig.setConfidenceThreshold(245)
        stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        camLeft.out.link(stereo.left)
        camRight.out.link(stereo.right)

        # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
        extended_disparity = True
        # Better accuracy for longer distance, fractional disparity 32-levels:
        subpixel = False
        # Better handling for occlusions:
        lr_check = True

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(lr_check)
        stereo.setExtendedDisparity(extended_disparity)
        stereo.setSubpixel(subpixel)

        self.depth_out = pipeline.create(dai.node.XLinkOut)
        self.depth_out.setStreamName("depth")
        stereo.depth.link(self.depth_out.input)
        # decreases latency
        self.depth_out.input.setQueueSize(1)
        self.depth_out.input.setBlocking(False)
        print("Pipeline creeeeeeeeeeeeeated.")
        return pipeline


    def extract_hands(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # print('hand landmarker result: {}'.format(result))
        self.detection_result = result
        if self.detection_result is not None:
            hands_preds = []
            hand_landmarks_list = self.detection_result.hand_landmarks
            hand_world_landmarks_list = self.detection_result.hand_world_landmarks
            handedness_list = self.detection_result.handedness

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                hand_world_landmarks = hand_world_landmarks_list[idx]
                handedness = handedness_list[idx]

                if len(hand_landmarks)>0 and self.depth_map is not None:
                    hand = HandPrediction(handedness, hand_landmarks, hand_world_landmarks, self.resolution, self.depth_map, self.stereoInference)
                    hands_preds.append(hand)
            self.hands_predictions = hands_preds

    def next_frame_video(self):
        ret, frame = self.video.read()
        self.frame = frame
        self.depth_map = self.depth_maps[self.current_frame_index]
        self.timestamp = self.timestamps[self.current_frame_index]
        self.current_frame_index += 1
        self.new_frame = True
        return ret, frame
    
    def next_frame_livestream(self):
        self.timestamp = time.time()
        d_frame = self.depthQ.get()
        r_frame = self.rgbQ.get()
        if d_frame is not None:
            frame = d_frame.getFrame()
            frame = cv2.resize(frame, self.cam_data['resolution'])
            # print(frame.shape)
            self.depth_map=frame

        if self.print_rgb_stereo_latency:
            now=dai.Clock.now()
            rgb_latency= (now - r_frame.getTimestamp()).total_seconds()*1000
            depth_latency= (now - d_frame.getTimestamp()).total_seconds()*1000
            print(f'rgb latency: {rgb_latency} ms, depth latency: {depth_latency} ms')
        
        if self.show_disparity:
            depthFrameColor = cv2.normalize(self.depth_map, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            cv2.imshow(f'depth {self.device_id}', depthFrameColor)

        if r_frame is not None:
            frame = r_frame.getCvFrame() 
            # cv2.imshow('raw_'+name, frame)
            # frame = cv2.resize(frame, self.cam_data['resolution']) 
            # frame=cv2.flip(frame,1)
            # print(frame.shape)
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            success = True
            self.frame = frame
            self.new_frame = True
        else:
            self.frame = None
        return success, frame

    def get_hands_video(self, frame):
        if frame is not None and self.new_frame:
        # if frame is not None and depthFrame is not None:
            # mp_frame = cv2.cvtColor(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            # mp_frame = cv2.cvtColor(cv2.flip(self.frame,1), cv2.COLOR_BGR2RGB) 
            mp_frame = cv2.flip(frame,1)
            # mp_frame=self.frame
            frame_timestamp_ms = round(self.timestamp*1000)
            mp_image = mp.Image(image_format=self.format, data=mp_frame)
            landmark_results = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            self.extract_hands(landmark_results, mp_image, frame_timestamp_ms)
            self.new_frame = False
        return self.hands_predictions
    
    def get_hands_live_stream(self, frame):
        if frame is not None and self.new_frame:
        # if frame is not None and depthFrame is not None:
            # mp_frame = cv2.cvtColor(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            # mp_frame = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)
            mp_frame = cv2.flip(frame,1)
            # mp_frame=self.frame
            frame_timestamp_ms = round(time.time()*1000)
            # print('frame_timestamp_ms', frame_timestamp_ms)
            mp_image = mp.Image(image_format=self.format, data=mp_frame)
            self.landmarker.detect_async(mp_image, frame_timestamp_ms)
            self.new_frame = False
        return self.hands_predictions
    
    def get_depth_map(self):
        return self.depth_map


class HandPrediction:
    def __init__(self, handedness, landmarks, world_landmarks, img_res, depth_map, stereo_inference) -> None:
        self.handedness = handedness
        self.normalized_landmarks = np.array([[1-l.x,l.y,l.z] for l in landmarks])
        # self.landmarks = np.array([[max(min(1-l.x,1.),0.)*img_res[0], max(min(l.y,1.),0.)*img_res[1], l.z] for l in landmarks])
        # self.normalized_landmarks = landmarks
        self.normalized_world_landmarks = world_landmarks
        # self.world_landmarks = np.array([[l.x*img_res[0], l.y*img_res[1], l.z] for l in world_landmarks])
        self.label = handedness[0].category_name.lower()
        self.position, self.roi = stereo_inference.calc_spatials(self.depth_point(), depth_map)
        # self.position = self.position/1000
        
    def depth_point(self):
        return self.normalized_landmarks[0,:]
    

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
    
def crop_to_rect(frame):
    height = frame.shape[0]
    width  = frame.shape[1]
    delta = int((width-height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta:width-delta]

def get_hand_detector( replay= False, replay_data = None, cam_params= None, device_id = None, detect_hands = True, resolution = (1280,720), fps = 30):

    hand_detector = HybridOAKMediapipeDetector(replay = replay, replay_data=replay_data, cam_params=cam_params, device_id=device_id, detect_hands=detect_hands, resolution=resolution, fps=fps, show_disparity=True, print_rgb_stereo_latency=True)
    return hand_detector