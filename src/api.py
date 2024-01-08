import cv2
import os
import json
import numpy as np
import base64
from detection import DriverDrowsiness as dd
from fastapi import FastAPI, File 

app = FastAPI(docs_url="/")

# Facial keypoints
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'config', 'config.json')
keypoints = json.load(open(file_path))
# Facial Landmarks Model
detect = dd(keypoints=keypoints)

@app.post("/detect/")    
async def detect_drowsiness(file: bytes = File(...)):
    image = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    result = detect.process_frame(image)
    try:
        frame, mor, ear, head = result
    except:
        frame = result
        mor = None
        ear = None
        head = None

    # Encode the image to base64
    _, frame_encode = cv2.imencode('.jpg', frame)
    byte_encode = base64.b64encode(frame_encode.tobytes()).decode('utf-8')

    return {
        "image": byte_encode,
        "mor": mor,
        "ear": ear,
        "head": head
    }
