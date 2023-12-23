import streamlit as st
import json
import cv2
import time
import pygame
from utils.drawing import Draw
from detection import DriverDrowsiness

def fps_count(cTime, pTime):
    return 1 / (cTime - pTime)

def get_path(options):
    if options == "IP_CAM":
        VIDEO_PATH = st.text_input(label="", placeholder="Enter your IP Camera address", type="password")
        VIDEO_PATH += "//video"
    elif options== "Webcam":
        VIDEO_PATH=0
    return VIDEO_PATH

def play_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load("src\Alarm\Alarm10.wav")
    pygame.mixer.music.play()

def main():
    # Streamlit app title and introduction
    st.title("Driver Drowsiness Detection")
    st.write("This app reads video data and processes it to detect drowsiness.")

    # Configurations
    keypoints = json.load(open('src\\config\\config.json'))
    options = st.radio("Choose Options", ["IP_CAM", "Webcam"], horizontal=True)
    VIDEO_PATH = get_path(options)
    col1, col2, _ = st.columns(spec=[0.1, 0.1, 0.8], gap = "small")
    with col1: start = st.button("\n :green[▶️]")
       
    if start:
        cap = cv2.VideoCapture(VIDEO_PATH)
        # Display the video stream in Streamlit
        video_placeholder = st.empty()
        # Processing loop
        if cap.isOpened():
            pTime = 0
            mor_threshold = 0.6
            ear_threshold = 0.25
            
            # frame durations 
            start_time = time.time()
            yawn_start_time = time.time()
            eye_close_start_time = time.time()
            
            # yawning constants 
            yawn_duration_threshold = 6 # this is the maximum limit of yawn in seconds
            yawn_count = 0 # counting yawns in a minute
            
            # eye blink constants
            eye_close_duration_threshold = 3 # if the eye is closed for 3 seconds, this keeps track
            EYE_AR_CONSEC_FRAMES = 2 # if the consecutive frames from ear is less than threshold then it considers it to be a blink
            eye_blinks = 0 # this counts total number of eye blinks in a minute
            blink_frame_counter = 0 # this counts number of blinks per frame
            
            # our detector instant for Driver Drowsiness
            drowsiness_detector = DriverDrowsiness(keypoints=keypoints)
            # our drawing instant for Drawing on frames
            draw = Draw()
            with col2: stop = st.button(":red[🟥]")
            # starting frame loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if stop:
                    # Release the captured frame
                    cap.release()
                    break
                
                # Processing frame
                results = drowsiness_detector.process_frame(frame)
                duration_threshold = (time.time() - start_time)>=60
                if duration_threshold:
                    start_time = time.time()
                    eye_blinks = 0
                    yawn_count = 0
                try:
                    result_frame, mor, ear, head_point = results
                    mouth_open = mor > mor_threshold
                    partial_eye_closure = ear < ear_threshold
                     
                    
                    # if the subject is yawning
                    if mouth_open:
                        # Calculate elapsed time yawning
                        elapsed_time = time.time() - yawn_start_time
                        # if the elapsed time exceeds our threshold
                        if elapsed_time >= yawn_duration_threshold:
                            # then we say our driver is yawning nad we count the yawn
                            yawn_count +=1
                    # if the driver stops yawning or doesn't yawn then we reset the yawn_start_time
                    else:
                        yawn_start_time = time.time()
                    
                    # now we check if the driver is exceeding the yawn limit within a minute
                    if yawn_count >= 3 and duration_threshold:
                        # now we warn the driver of his sleepiness
                        draw.draw_sleepy(result_frame, head_point)
                        play_alarm()
                        yawn_count = 0
                        
                    else:
                        pass
                    
                    # eye blink monitoring    
                    # if the driver is blinking
                    if partial_eye_closure:
                        blink_frame_counter+=1
                        # we count elapsed time of closed eyes
                        elapsed_time = time.time() - eye_close_start_time
                        # if the eyes are closed for too long, say 3 seconds
                        if elapsed_time >= eye_close_duration_threshold:
                            # then we alert the driver
                            draw.draw_sleepy(result_frame, head_point)
                            play_alarm()
                    else:
                        if blink_frame_counter > EYE_AR_CONSEC_FRAMES:
                            eye_blinks+=1
                        eye_close_start_time = time.time()
                        blink_frame_counter = 0
                    # when eye blinks are between 5 to 5 per minute, the driver is considered drowsy
                    if 5<=eye_blinks<=6 and duration_threshold:
                        draw.draw_sleepy(result_frame, head_point)
                        play_alarm()
                        eye_blinks = 0
                    elif (8<=eye_blinks or eye_blinks>=21) and not duration_threshold:
                        draw.draw_stress_driving(result_frame, eye_blinks)
                    
                    # Display the processed frame in Streamlit
                    video_placeholder.image(result_frame, channels="BGR", use_column_width=True, output_format="JPEG")

                except:
                    result_frame = results
                
                finally:
                    # FPS count
                    cTime = time.time()
                    fps = fps_count(cTime, pTime)
                    pTime = cTime
                    draw.draw_fps_count(result_frame, fps)
                    draw.draw_blink_count(result_frame, eye_blinks)
                    
                    # Display the processed frame in Streamlit
                    video_placeholder.image(result_frame, channels="BGR", use_column_width=True, output_format="JPEG")

        else:
            st.error("Error: Could not open video")
    else:
        st.markdown("Press :green[▶️] to Detect Driver Drowsiness")
# Run the Streamlit app
if __name__ == "__main__":
    main()
