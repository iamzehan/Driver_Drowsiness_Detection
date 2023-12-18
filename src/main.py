import json
import cv2 
import time
from utils.drawing import Draw
from detection import DriverDrowsiness

"""
This file is responsible for reading video data and processing them to detect drowsiness.
"""
if __name__ == "__main__":
    
    # reading the config file to get camera path
    config_data = json.load(open("config.json", "r"))
    keypoints = json.load(open('src\\config\\config.json'))
    # getting the path from the file
    VIDEO_PATH = config_data['IP_CAM']["phone"]
    # VIDEO_PATH = "Videos/3.mp4"
    # capture the video 
    cap = cv2.VideoCapture(VIDEO_PATH)

    #feeding the frames in a loop
    if cap.isOpened():
        pTime = 0
        try:
            detector = DriverDrowsiness(keypoints=keypoints)
            draw = Draw()
        except Exception as e:
            print({e})
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result_frame = detector.process_frame(frame)
            
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.startWindowThread()
            draw.draw_fps_count(result_frame,fps)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # if the video doesn't open
    else:
        print("Error: Could not open video")
    
    # release the captured frame
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Needed to close the imshow window