# Pose Analysis 

## Overview
This project is designed to perform pose analysis on any image or video file. The primary goal is to detect persons in the video, extract their poses, and save the results in a structured JSON format. The project leverages pre-trained models for both detection and pose estimation, using the MMDetection and MMPose libraries.

## Project Structure
```
PoseAnalysisProject/
│
├── config_files/            # Contains configuration files and model weights
├── data/                    # Input data (video/image files)
├── output/                  # Output directory for JSON results
├── .gitignore               # Git ignore file
├── README.md                # Documentation file
├── config.json              # JSON configuration file
├── creating_config.py       # Script to create configuration JSON
├── detection_service.py     # Detection service script
├── pose_analysis_service.py # Pose analysis service script
└── pose_extraction_service.py# Pose extraction service script
```

## Installation

To get started, you need to install the necessary dependencies. Follow the installation guide provided by MMPose:

[MMPose Installation](https://mmpose.readthedocs.io/en/latest/installation.html)

### Download model weights

Before running the project, download the pre-trained model weights:

1. Detection Model:  
    Download the person detection model weights from [this link](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth) and save it in the `config_files` folder.

2. Pose Extraction Model:  
    Download the pose extraction model weights from [this link](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth) and save it in the `config_files` folder.

## Configuration
The project uses a JSON configuration file (`config.json`) to manage various settings such as model paths, model weights, input/output file locations, and processing thresholds.

You can create or modify the configuration file using the `creating_config.py` script.

## Running the Project
1. **Pose Analysis Service:**  
    The main entry point for this project is the `pose_analysis_service.py` script. This script initializes the detection and pose extraction services, processes the input image or video, and saves the results as a JSON file.

    The script reads the configuration from the `config.json` file. It initializes the DetectionService and PoseExtractionService using the provided model paths and parameters.It processes the input image or  video frame by frame, detecting persons and extracting their poses. The results are saved in the specified output JSON file.
    
    To run the analysis:
    ```sh
    python pose_analysis_service.py
    ```

    This will process the input image or video specified in the configuration and generate a pose analysis output saved in the path specified in the configuration file.

2. **Detection Service:**  
    The `detection_service.py` script provides a service to detect persons in an image or video frame using a pre-trained detection model.

    The script initializes the detection model with the specified configuration and model checkpoint. It runs inference on the input image or frame to detect bounding boxes for persons. The bounding boxes are filtered based on the provided thresholds and returned for further processing.

3. **Pose Extraction Service:**  
    The `pose_extraction_service.py` script provides a service to extract poses from detected persons using a pre-trained pose estimation model.

    The script initializes the pose estimation model with the specified configuration and model checkpoint. It processes the detected bounding boxes in an image to estimate the poses of the persons within the boxes. The pose data is returned for integration into the final output JSON.


## Output
The output of the pose analysis is saved in a JSON file in the output directory. The JSON file contains detailed information about the detected poses, including keypoint coordinates, confidence scores, and other metadata.