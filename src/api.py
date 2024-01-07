from fastapi import FastAPI
from pydantic import BaseModel
from detection import DriverDrowsiness
import numpy as np
import json

app = FastAPI(docs_url="/")

keypoints=json.load(open('src\\config\\config.json'))
detect = DriverDrowsiness(keypoints=keypoints)

class Payload(BaseModel):
    image: np.array
    
@app.post("/detect/")
async def get_flower_class(file: Payload.image):
    result = await detect.process_frame(file.image)
    if len(result)>1:
        frame, mor, ear, head = result
        return {
                "frame":frame,
                "mor": mor,
                "ear": ear,
                "head":head
                }
    else:
        return {
            "frame": result
            }
