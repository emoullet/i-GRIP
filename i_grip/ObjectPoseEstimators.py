#!/usr/bin/env python3
import torch
import os
import cv2
import numpy as np
import pandas as pd
from i_grip.image_utils import make_cameras
from i_grip.model_utils import load_pose_predictor
from cosypose.utils.tensor_collection import PandasTensorCollection
from i_grip.config import TLESS_COARSE_ESTIMATOR_ID, TLESS_REFINER_ESTIMATOR_ID, YCVB_COARSE_ESTIMATOR_ID, YCVB_REFINER_ESTIMATOR_ID


class KnownObjectPoseEstimator:
    def __init__(self, dataset, cam_data, render_txt =False, render_overlay = False, render_bboxes = True, 
                    use_tracking = True, fuse_detections = True):
        print('Building KnownObjectPoseEstimator')
        cam_mat = cam_data['matrix']
        img_resolution = cam_data['resolution']
        self.render_txt = render_txt
        self.render_overlay = render_overlay
        self.render_bboxes = render_bboxes
        self.img_ratio = 4/3
        # Prepare camera infos
        intrinsics = dict(
            fx=cam_mat[0,0], cx=cam_mat[0,2],
            fy=cam_mat[1,1], cy=cam_mat[1,2],
            resolution=img_resolution,
        )
        print(f'Intrinsics: {intrinsics}')
        self.dataset = dataset
        #self.windows = [((0,0),(1152,648))]

        if(dataset == "ycbv"):
            object_coarse_run_id = YCVB_COARSE_ESTIMATOR_ID
            object_refiner_run_id = YCVB_REFINER_ESTIMATOR_ID
            # object_coarse_run_id = 'coarse-bop-ycbv-pbr--724183'
            # object_refiner_run_id = 'refiner-bop-ycbv-pbr--604090'
        elif(dataset == "tless"):
            object_coarse_run_id = TLESS_COARSE_ESTIMATOR_ID
            object_refiner_run_id = TLESS_REFINER_ESTIMATOR_ID
        else:
            assert False
        print('object_coarse_run_id',object_coarse_run_id)
        print('object_refiner_run_id',object_refiner_run_id)
        self.debug_converter = None
        self.cameras = make_cameras([intrinsics])
        self.pose_predictor = load_pose_predictor(object_coarse_run_id,
                                                  object_refiner_run_id,
                                                  preload_cache=True,
                                                  n_workers=6 )
        self.use_prior = use_tracking
        self.threshold_nb_iter = 20

        self.K = self.cameras.K.cuda().float()
        self.n_refiner_iterations = 1
        # self.emptyPrediction = KnownObjectPoseEstimator.emptyPrediction()

        self.scene_objects = dict()
        self.fuse_detections = fuse_detections
        if self.fuse_detections:
            self.predict = self.pose_predictor.get_predictions_fused
        else:
            self.predict = self.pose_predictor.get_predictions
        self.reset()
        print('KnownObjectPoseEstimator built')
        
    def reset(self):
        self.pose_estimation_prior = None
        self.pose_predictions = None
        self.it = 0
    
    def estimate(self, image, detections = None):
        if image is None:
            return {}
        # Predict poses using cosypose
        #print(detections)
        # print(f'pose_estimation_prior {self.pose_estimation_prior}')
        # print(f'detections {detections.bboxes}')
        predict = self.pose_estimation_prior is not None or detections is not None
        if predict:
            # image = torch.as_tensor(np.stack([self.format_crop(image), ])).permute(0, 3, 1, 2).cuda().float() / 255
            img = torch.as_tensor(np.stack([image, ])).permute(0, 3, 1, 2).cuda().float() / 255
            # cv2.imwrite('/home/emoullet/GitHub/i-GRIP/tset/img.png', image)
            # save detections.bboxes as file
            # np.savetxt('/home/emoullet/GitHub/i-GRIP/tset/detections.txt', detections.copy().cpu().bboxes)
            # save K as file
            # print(f'K {self.K}')
            if detections is not None and not self.fuse_detections:
            #if detections is not None:
                self.pose_estimation_prior = None
                # print('NEW DETECTION')
            else:
                # print('TRACKING')
                pass

            if self.pose_estimation_prior is None or self.fuse_detections:
                n_coarse_iterations=1
            else:
                n_coarse_iterations = 0
            # inps = dict(
            #     images=img, K=self.K,
            #     data_TCO_init=self.pose_estimation_prior,
            #     n_coarse_iterations=n_coarse_iterations,
            #     n_refiner_iterations=self.n_refiner_iterations,
            #     detections=detections)
            # print(f'inps : {inps}')
            self.pose_predictions, _ = self.predict(
                images=img, K=self.K,
                data_TCO_init=self.pose_estimation_prior,
                n_coarse_iterations=n_coarse_iterations,
                n_refiner_iterations=self.n_refiner_iterations,
                detections=detections
            )
            # print('ITER')
            # print(self.pose_predictions.poses)
            # print(self.pose_predictions.poses_input)
            # print(self.pose_predictions.K_crop)
            # print(self.pose_predictions.boxes_rend)
            # print(self.pose_predictions.boxes_crop)
            # exit()
        else:
                self.pose_predictions = None

        if self.use_prior :
            self.pose_estimation_prior = self.pose_predictions
        else:
            self.pose_estimation_prior = None

            # if self.nb_iter_without_detection <= self.threshold_nb_iter:
            #     self.nb_iter_without_detection +=1
            # else:
            #     self.nb_iter_without_detection = 0
            #     self.pose_estimation_prior = None
            #     self.detect = True          
        objects_predictions = {}
        if self.pose_predictions is not None:  
            for n in range(len(self.pose_predictions)):
                label=self.pose_predictions.infos.loc[n, 'label']
                pose=self.pose_predictions.poses[n]
                render_box=self.pose_predictions.boxes_rend[n]                
                score=self.pose_predictions.infos['score'][n]
                objects_predictions[label] = ObjectPoseEstimation(label, pose, render_box, score, self.dataset)
        return objects_predictions
    
    def format(self, img):
        padded_img = cv2.copyMakeBorder(img, 0,self.pad_h,0,0,cv2.BORDER_CONSTANT)
        print(padded_img.shape)
        return padded_img
    
    def format_crop(self, img):
        res=(640, 480)
        croped = img[:res[0],:res[1]]
        return croped


    def stop(self):
        self.pose_predictor.stop()
        pass

    def emptyPrediction():
        return PandasTensorCollection(infos=pd.DataFrame(dict(label=[],)),
                                      poses=torch.empty((0, 4, 4)).float().cuda())
class ObjectPoseEstimation:
    def __init__(self, label , pose, render_box = None, score = None, dataset=None) -> None:
        self.label = label
        self.pose = pose
        self.render_box = render_box
        self.score = score
        self.dataset = dataset
        
def get_pose_estimator(dataset, cam_data, use_tracking = True, fuse_detections = False):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    detector = KnownObjectPoseEstimator(dataset, cam_data, use_tracking=use_tracking, fuse_detections=fuse_detections)
    return detector

if __name__ == '__main__':
    pass