from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
import io
from PIL import Image
import os
import uuid
import datetime
import cv2
import numpy as np

app = FastAPI()

# Load your trained model globally to avoid reloading on each API request
try:
    model = YOLO("best.pt") # Ensure 'best.pt' is in the same directory as app.py
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None # Set model to None if loading fails

# Placeholders for digital twin logging and user states
activity_logs = []
user_states = {} # Example: {user_id: {"level": "beginner"}}

@app.get("/")
async def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "YOLO Object Detection API is running!", "model_loaded": model is not None}

@app.post("/detect/")
async def detect_object(file: UploadFile = File(...), user_id: str = "anonymous"):
    """
    API endpoint to perform object detection on an uploaded image.
    Also logs user actions for digital twin compatibility.
    """
    session_id = str(uuid.uuid4()) # Use unique session IDs (UUIDs)
    timestamp = datetime.datetime.now().isoformat() # Every user action must be logged with time-based identifiers
    action_metadata = {"filename": file.filename, "content_type": file.content_type}

    # Log the user action
    activity_logs.append({
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "action": "image_upload_for_detection",
        "metadata": action_metadata
    })
    print(f"Logged action: {activity_logs[-1]}")

    if model is None:
        activity_logs[-1]["result"] = {"status": "failure", "error": "Model not loaded."}
        return JSONResponse(content={"error": "Model not loaded. Check server logs."}, status_code=500)

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        results = model(image, verbose=False) # Perform inference, verbose=False for cleaner logs

        detections = []
        for r in results:
            # 'box' attribute holds bounding boxes, confidence, and class
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                label = model.names[cls]
                detections.append({
                    "class": label,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })

        # Log the result of the action
        activity_logs[-1]["result"] = {"status": "success", "detections_count": len(detections)}
        print(f"Logged result: {activity_logs[-1]}")

        return JSONResponse(content={"filename": file.filename, "detections": detections})

    except Exception as e:
        # Log failure
        activity_logs[-1]["result"] = {"status": "failure", "error": str(e)}
        print(f"Logged error result: {activity_logs[-1]}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/detect_video/")
async def detect_video(file: UploadFile = File(...), user_id: str = "anonymous"):
    """
    API endpoint to perform object detection on an uploaded video.
    Processes frame-by-frame and saves an annotated output video.
    Logs user actions for digital twin compatibility.
    """
    session_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    action_metadata = {"filename": file.filename, "content_type": file.content_type}

    activity_logs.append({ # Log the action
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "action": "video_upload_for_detection",
        "metadata": action_metadata
    })
    print(f"Logged action: {activity_logs[-1]}")

    if model is None:
        activity_logs[-1]["result"] = {"status": "failure", "error": "Model not loaded."}
        return JSONResponse(content={"error": "Model not loaded. Check server logs."}, status_code=500)

    temp_video_path = None
    output_video_path = None
    try:
        # 1. Save the uploaded video to a temporary file inside the container
        temp_video_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())

        # 2. Define output video path
        output_video_filename = f"detected_{uuid.uuid4()}_{file.filename}"
        output_video_path = f"/tmp/{output_video_filename}"

        # 3. Read video using OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 4. Create VideoWriter object to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4. Use '*XVID' for .avi if mp4v fails.
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read() # Read a frame
            if not ret:
                break # End of video

            frame_count += 1

            # 5. Perform YOLO inference on the frame
            results = model(frame, verbose=False) # verbose=False for cleaner logs

            # 6. Draw bounding boxes/labels on the frame using Ultralytics' plot method
            annotated_frame = results[0].plot() # Gets the frame with detections drawn

            # 7. Write the annotated frame to the output video file
            if annotated_frame is not None:
                out.write(annotated_frame)
            else: # Fallback if plot() returns None
                out.write(frame)

        # 8. Release video capture and writer objects
        cap.release()
        out.release()

        # 9. Clean up temporary input video file
        os.remove(temp_video_path)

        # 10. Log the result
        activity_logs[-1]["result"] = {"status": "success", "output_video": output_video_filename, "total_frames": frame_count}
        print(f"Logged result: {activity_logs[-1]}")

        # 11. Return response (In production, you'd usually return a URL to download the video)
        return JSONResponse(content={
            "filename": file.filename,
            "message": "Video processed successfully! Output saved temporarily in container.",
            "output_video_filename": output_video_filename,
            "total_frames_processed": frame_count
        })

    except Exception as e:
        # Log failure
        activity_logs[-1]["result"] = {"status": "failure", "error": str(e)}
        print(f"Logged error result: {activity_logs[-1]}")
        # Clean up temp files if an error occurs
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if output_video_path and os.path.exists(output_video_path):
            os.remove(output_video_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/logs")
async def get_logs():
    """Endpoint to retrieve activity logs (as JSON dump)."""
    # All logs should be available as JSON dumps
    return JSONResponse(content=activity_logs)

@app.get("/simulate_user/")
async def simulate_user_dummy():
    """
    Dummy simulation route as per digital twin requirements.
    Returns mock prediction JSON.
    """
    mock_prediction = {
        "risk": "low",
        "recommended_action": "continue",
        "simulation_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat()
    }
    # Log the simulation event as well
    activity_logs.append({ # All agent actions must be logged: input received, action taken, result.
        "user_id": "system_simulation",
        "session_id": mock_prediction["simulation_id"],
        "timestamp": mock_prediction["timestamp"],
        "action": "user_simulation_run",
        "metadata": {"prediction": mock_prediction}
    })
    return JSONResponse(content=mock_prediction)

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)