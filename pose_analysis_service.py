import cv2
import json
import numpy as np
import mimetypes
from detection_service import DetectionService
from pose_extraction_service import PoseExtractionService
from mmpose.structures import split_instances

class PoseAnalysisService:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as json_file:  
            self.config = json.load(json_file)

        # Initialize services
        self.detector = DetectionService(
            self.config
        )
        self.pose_extractor = PoseExtractionService(
            self.config
        )
        self.input = self.config["input_file"]
        self.output_json = self.config["output_json"] 
    
    
    def process_one_image(self, img, visualizer=None, show_interval=0):
        bboxes = self.detector.detect(img)
        pred_instances = self.pose_extractor.extract_pose(img, bboxes)
        return pred_instances 

    def analyze(self):
        input_type = mimetypes.guess_type(self.input)[0].split('/')[0]
        if input_type == 'image':
            pred_instances = self.process_one_image(self.input)
            pred_instances_list = split_instances(pred_instances)
            for i in range(len(pred_instances_list)):
                pred_instances_list[i]['bbox_score'] = pred_instances_list[i]['bbox_score'].item()
            
        elif input_type == 'video':
            cap = cv2.VideoCapture(self.input)
            pred_instances_list = []
            video_writer = None
            frame_idx = 0
            while cap.isOpened():
                success, frame = cap.read()
                frame_idx += 1

                if not success:
                    break

                # topdown pose estimation
                pred_instances = self.process_one_image(frame)
                
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))
            cap.release()
            for i in range(len(pred_instances_list)):
                for j in range(len(pred_instances_list[i]['instances'])):
                    pred_instances_list[i]['instances'][j]['bbox_score'] = pred_instances_list[i]['instances'][j]['bbox_score'].item()
        
        ndarray_list = ['keypoint_colors', 'skeleton_link_colors', 'dataset_keypoint_weights', 'sigmas']
        for key in ndarray_list:
            self.pose_extractor.pose_estimator.dataset_meta[key] = self.pose_extractor.pose_estimator.dataset_meta[key].tolist()
        
        with open(self.output_json, 'w') as f:
            json.dump(
                dict(
                    meta_info=self.pose_extractor.pose_estimator.dataset_meta,
                    instance_info=pred_instances_list
                    ),
                f,
                indent='\t')
        print(f'predictions have been saved at {self.output_json}')

def main():
    json_path = "config.json"
    pose_analysis = PoseAnalysisService(json_path)
    pose_analysis.analyze()

if __name__ == "__main__":
    main()