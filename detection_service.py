import json
import numpy as np 
from mmdet.apis import init_detector, inference_detector
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline

class DetectionService:
    def __init__(self, config):
        # configurations
        self.config = config
        device = self.config["device"]
        
        self.detector = init_detector(
            self.config["det_config"],
            self.config["det_checkpoint"],
            device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

    def detect(self, image):
        # configurations
        det_cat_id = self.config["det_cat_id"]
        bbox_thr = self.config["bbox_thr"]
        nms_thr = self.config["nms_thr"]
        
        # predict bbox
        det_result = inference_detector(self.detector, image)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == det_cat_id,
                                    pred_instance.scores > bbox_thr)]
        bboxes = bboxes[nms(bboxes, nms_thr), :4]
        
        return bboxes 
