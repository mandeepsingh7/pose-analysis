import json 

# Define the configuration as a dictionary
config = {
    "det_config": "config_files/rtmdet_m_640-8xb32_coco-person.py",
    "det_checkpoint": "config_files/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
    "pose_config": "config_files/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py",
    "pose_checkpoint": "config_files/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth",
    "device": "cpu",  # Example: "cuda:0" for GPU, "cpu" for CPU
    "det_cat_id": 0,  # category ID for detection
    "bbox_thr": 0.3,  # bounding box threshold
    "nms_thr": 0.3,    # non-maximum suppression threshold
    "input_file": "data/racket_0_1s.mp4", # Input image or video 
    "output_json": "output/output_file_1.json" # Output path of JSON file 
}

json_path = 'config.json'

with open(json_path, "w") as json_file:
    json.dump(config, json_file, indent=4)