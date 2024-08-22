import json
import numpy as np
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples

class PoseExtractionService:
    def __init__(self, config):
        self.config = config    
        # configurations
        device = self.config["device"]
        
        self.pose_estimator = init_pose_estimator(
            self.config["pose_config"],
            self.config["pose_checkpoint"],
            device=device)

    def extract_pose(self, image, bboxes):
        # Run pose estimation
        pose_results = inference_topdown(
            self.pose_estimator,
            image,
            bboxes
        )
        data_samples = merge_data_samples(pose_results)
        return data_samples.get('pred_instances', None)