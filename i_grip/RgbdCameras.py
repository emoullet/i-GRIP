import depthai as dai
import cv2
import numpy as np
import time

class RgbdCamera:
    
    _720P = [1280., 720.]
    _1080P = [1920., 1080.]
    _480P = [640., 480.]
    _480P = [640., 360.]
    _RGB_MODE = 'RGB'
    _BGR_MODE = 'BGR'
    
    def __init__(self, replay = False, 
                 replay_data= None, 
                 cam_params = None, 
                 device_id = None, 
                 fps=30., 
                 resolution = _480P, 
                 print_rgb_stereo_latency = False, 
                 show_disparity=False,
                 color_mode = _BGR_MODE) -> None:
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
        print('Building RGBd Camera...')
        self.cam_auto_mode = True
        self.device_id = device_id
        print(f'device_id: {self.device_id}')
        self.replay = replay
        self.fps = fps
        self.resolution = resolution
        self.print_rgb_stereo_latency=print_rgb_stereo_latency
        self.show_disparity=show_disparity
        if color_mode not in [self._RGB_MODE, self._BGR_MODE]:
            raise ValueError(f'color_mode must be one of {self._RGB_MODE} or {self._BGR_MODE}')
        self.color_mode = color_mode
        
        print(f'fps: {fps}, resolution: {resolution}')
        if self.replay :
            if replay_data is not None:
                self.load_replay(replay_data)
            self.next_frame = self.next_frame_video
            self.is_on = self.is_on_replay
            self.cam_data = cam_params
        else:
            self.cam_data = {}
            self.build_device()
            self.next_frame = self.next_frame_livestream

        self.frame = None
        self.new_frame = False

        print(f'RGBd Camera built: replay={replay}')
    

    def build_device(self):
        print("Building device...")
        print(f'device_id: {self.device_id}')
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


    def load_replay(self, replay):
        self.video = cv2.VideoCapture(replay['Video'])
        self.nb_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0
        if not self.video.isOpened():
            print("Error reading video") 
            exit()
        print(replay.keys())
        self.timestamps = replay['Timestamps']
        self.depth_maps = replay['Depth_maps']
        
    def next_frame_livestream(self):
        self.timestamp = time.time()
        d_frame = self.depthQ.get()
        r_frame = self.rgbQ.get()
        if d_frame is not None:
            frame = d_frame.getFrame()
            frame = cv2.resize(frame, self.cam_data['resolution'])
            self.depth_map = frame
        else:
            self.depth_map = None

        if self.print_rgb_stereo_latency:
            now = dai.Clock.now()
            rgb_latency = (now - r_frame.getTimestamp()).total_seconds() * 1000
            depth_latency = (now - d_frame.getTimestamp()).total_seconds() * 1000
            print(f'rgb latency: {rgb_latency} ms, depth latency: {depth_latency} ms')

        if self.show_disparity:
            depthFrameColor = cv2.normalize(self.depth_map, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            cv2.imshow(f'depth {self.device_id}', depthFrameColor)

        if r_frame is not None:
            frame = r_frame.getCvFrame()
            frame = cv2.resize(frame, self.cam_data['resolution'])
            if self.color_mode == self._RGB_MODE:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame = frame
            self.new_frame = True
        else:
            self.frame = None
            
        if self.frame is not None and self.depth_map is not None:
            success = True
            return success, self.frame, self.depth_map
        else:
            return False, None, None
    
    
    def next_frame_video(self):
        success, frame = self.video.read()
        # frame = cv2.resize(frame, self.cam_data['resolution'])
        self.frame = frame
        self.depth_map = self.depth_maps[self.current_frame_index]
        # self.depth_map = cv2.resize(self.depth_map, self.cam_data['resolution'])
        self.timestamp = self.timestamps[self.current_frame_index]
        self.current_frame_index += 1
        self.new_frame = True
        return success, self.frame, self.depth_map
    
    def get_depth_map(self):
        return self.depth_map

    def start(self):
        self.on = True
    
    def stop(self):
        self.on = False
        #wait for 50ms
        if not self.replay:
            time.sleep(0.05)
            self.device.close()

    def get_res(self):
        return self.cam_data['resolution']
    
    def get_device_data(self):
        return self.cam_data

    def get_num_frames(self):
        return self.nb_frames
    
    def get_timestamps(self):
        return self.timestamps
    
    
    def is_on(self):
        return self.on
    
    def is_on_replay(self):
        return self.current_frame_index<self.nb_frames
