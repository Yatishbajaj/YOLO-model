# YOLO-model
Drone detection syatem
# YOLO Object Detection API with FastAPI and Docker
This repository contains a real-time object detection API built using FastAPI and the Ultralytics YOLOv8 model. It allows you to perform object detection on both images and videos, with logging capabilities designed for digital twin integration. The application is containerized using Docker for easy deployment.

# Table of Contents
Features

Project Structure

Prerequisites

Setup and Installation

1. Clone the Repository

2. Prepare Your YOLO Model

3. Build and Run with Docker

4. Running Locally (Without Docker)

API Endpoints

Digital Twin Logging

Training Your Own Model

Contributing

# Features
Real-time Object Detection: Detects objects in images and videos using a pre-trained YOLOv8 model.

FastAPI Backend: A high-performance Python web API.

Video Processing: Processes video files frame-by-frame, applies detections, and saves the annotated output.

Digital Twin Logging: Automatically logs user actions and detection results for future analysis and digital twin compatibility.

Dockerized Deployment: Easy to build and run the application in a consistent environment.

Custom Model Support: Designed to work with your own trained best.pt YOLO model.

# Project Structure
.
├── app.py              # FastAPI application with detection logic and API endpoints
├── dockerfile          # Dockerfile for building the application image
├── main.py             # Script for training the YOLO model (example)
├── requirements.txt    # Python dependencies
└── best.pt             # Your pre-trained YOLO model weights (place this file here)

# Prerequisites
Before you begin, ensure you have the following installed:

Docker (recommended for easy setup)

Python 3.9+ (if running locally)

pip (Python package installer)

# Setup and Installation
1. Clone the Repository
git clone <your-repository-url>
cd <your-repository-name>

# 2. Prepare Your YOLO Model
This project expects a trained YOLO model named best.pt in the root directory. If you have trained your own YOLOv8 model, place its best.pt file here. If you don't have one, you can use a pre-trained YOLOv8n model (e.g., download from Ultralytics GitHub releases and rename it to best.pt).

Important: The best.pt file must be in the same directory as app.py and dockerfile.

# 3. Build and Run with Docker (Recommended)
This is the easiest way to get the API running, as it handles all dependencies.

Create requirements.txt:
Make sure you have a requirements.txt file in the root directory with the necessary Python packages. Here's a basic requirements.txt for this project:

fastapi
uvicorn[standard]
ultralytics
Pillow
python-multipart
opencv-python-headless # Use headless for server environments without GUI
numpy

Build the Docker Image:
Navigate to the project root directory in your terminal and run:

docker build -t yolo-detection-api .

This command builds a Docker image named yolo-detection-api. It copies your best.pt and app.py into the image.

Run the Docker Container:
Once the image is built, you can run the container:

docker run -p 8000:8000 yolo-detection-api

This maps port 8000 on your host machine to port 8000 inside the container, where the FastAPI application runs.

The API should now be accessible at http://localhost:8000.

# 4. Running Locally (Without Docker)
If you prefer to run the application directly on your machine:

Install Dependencies:
First, ensure you have the requirements.txt file as described above. Then install the dependencies:

pip install -r requirements.txt

Run the FastAPI Application:
Navigate to the project root directory and run:

uvicorn app:app --host 0.0.0.0 --port 8000

The API will be available at http://localhost:8000.

# API Endpoints
Once the server is running, you can access the API documentation (Swagger UI) at http://localhost:8000/docs.

GET /

A simple endpoint to check if the API is running and if the YOLO model is loaded.

Response: {"message": "YOLO Object Detection API is running!", "model_loaded": true/false}

POST /detect/

Performs object detection on an uploaded image.

Method: POST

Parameters:

file: The image file to upload (UploadFile).

user_id: (Optional) A string to identify the user for logging purposes. Defaults to "anonymous".

Response: JSON containing filename and a list of detected objects (class, confidence, bounding box coordinates).

POST /detect_video/

Performs object detection on an uploaded video, processes it frame-by-frame, and saves an annotated output video temporarily.

Method: POST

Parameters:

file: The video file to upload (UploadFile).

user_id: (Optional) A string to identify the user for logging purposes. Defaults to "anonymous".

Response: JSON indicating success and the filename of the processed video (saved within the container's /tmp directory). Note: In a production environment, you would typically implement a mechanism to serve or download this video.

GET /logs

Retrieves all recorded activity logs in JSON format.

Response: A JSON array of log entries, each containing user_id, session_id, timestamp, action, and metadata.

GET /simulate_user/

A dummy endpoint for simulating user actions, primarily for digital twin integration testing.

Response: Mock prediction JSON.

# Digital Twin Logging
The app.py includes built-in logging for user actions and simulation events. Every image upload, video upload, and dummy simulation run is logged with a user_id, session_id (UUID), timestamp, action, and relevant metadata. These logs can be retrieved via the /logs endpoint, making the API compatible with digital twin requirements for tracking agent actions and user journey states.

# Training Your Own Model
The main.py script is provided as an example for training your own YOLOv8 model using the Ultralytics library.

To use main.py:

Prepare your dataset: Ensure your dataset is organized in YOLO format and you have a data.yaml file (e.g., data/KIIT-MiTA/KIIT-MiTA.yaml as shown in the example).

Install Ultralytics: Make sure ultralytics is installed (pip install ultralytics).

Run the training script:

python main.py

This script will train a YOLOv8 model for 50 epochs and save the best.pt weights in a runs/detect/trainX/weights/ directory. You would then copy this best.pt file to the root of your API project.

# Contributing
Feel free to fork this repository, make improvements, and submit pull requests.
