import json
import cv2 
import time
from utils.drawing import Draw
from detection import DriverDrowsiness

"""
This file is responsible for reading video data and processing them to detect drowsiness.
"""
def fps_count(cTime, pTime):
    return 1 / (cTime - pTime)

    
    
if __name__ == "__main__":
    
    # reading the config file to get camera path
    config_data = json.load(open("config.json", "r"))
    keypoints = json.load(open('src\\config\\config.json'))
    
    # getting the path from the file
    VIDEO_PATH = config_data['IP_CAM']["phone"]
    
    # capture the video 
    cap = cv2.VideoCapture(VIDEO_PATH)
 
    #feeding the frames in a loop
    if cap.isOpened():
        pTime = 0
        detector = DriverDrowsiness(keypoints=keypoints)
        draw = Draw()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result_frame = detector.process_frame(frame)
            cTime = time.time()
            fps = fps_count(cTime, pTime)
            pTime = cTime
            draw.draw_fps_count(result_frame, fps)
            cv2.startWindowThread()
            cv2.imshow("Output", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 break
            
    # if the video doesn't open
    else:
        print("Error: Could not open video")
    
    # release the captured frame
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Needed to close the imshow window