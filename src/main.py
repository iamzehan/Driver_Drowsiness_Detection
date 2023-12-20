import json
import cv2 
import time
import pygame
from utils.drawing import Draw
from detection import DriverDrowsiness

"""
This file is responsible for reading video data and processing them to detect drowsiness.
"""
def fps_count(cTime, pTime):
    return 1 / (cTime - pTime)

def play_alarm():
    pygame.mixer.init()
    # Load an alarm sound (replace "your_alarm_sound.wav" with the path to your sound file)
    pygame.mixer.music.load("src\Alarm\Alarm10.wav")
    pygame.mixer.music.play()
    
if __name__ == "__main__":
    
    # reading the config file to get camera path
    config_data = json.load(open("config.json", "r"))
    keypoints = json.load(open('src\\config\\config.json'))
    
    # getting the path from the file
    VIDEO_PATH = "http://192.168.0.159:8080//video"
    # capture the video 
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    #feeding the frames in a loop
    if cap.isOpened():
        pTime = 0
        duration_threshold = 4
        mor_threshold = 0.6
        ear_threshold = 0.25
        start_time = time.time()
        sleep_count = 0
        detector = DriverDrowsiness(keypoints=keypoints)
        draw = Draw()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # getting the results
            results = detector.process_frame(frame)
            try:
                result_frame, mor, ear, head_point = results
                #fps count
                cTime = time.time()
                fps = fps_count(cTime, pTime)
                pTime = cTime
                draw.draw_fps_count(result_frame, fps)
                
                # Alarting based on elapsed sleep time
                isSleepy = ( mor > mor_threshold or ear < ear_threshold)
                if isSleepy:
                    # calculate elapsed time sleeping and alert the driver
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    sleep_count+=1
                    if elapsed_time >= duration_threshold:
                        draw.draw_sleepy(result_frame, head_point)
                        play_alarm()
                        
                else:
                    # Reset the counters
                    start_time = time.time()
                    sleep_count = 0
            except:
                result_frame = results
                #fps count
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