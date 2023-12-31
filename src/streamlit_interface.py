import cv2
import json
import time
import pygame
import numpy as np
import datetime as dt
import streamlit as st
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
    pygame.mixer.music.play(1)

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
            yawn_frame_counter = 0
            MOR_CONSEC_FRAMES = 90
            
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
                        yawn_frame_counter += 1
                        
                    # if the driver stops yawning or doesn't yawn then we reset the yawn_start_time
                    else:
                        if yawn_frame_counter >= MOR_CONSEC_FRAMES:
                            yawn_count+=1
                        yawn_frame_counter = 0
                    
                    # now we check if the driver is exceeding the yawn limit within a minute
                    if not mouth_open and yawn_count >= 2 and duration_threshold:
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
                    draw.draw_yawn_count(result_frame, yawn_count)
                    
                    # Display the processed frame in Streamlit
                    video_placeholder.image(result_frame, channels="BGR", use_column_width=True, output_format="JPEG")

        else:
            st.error("Error: Could not open video")
    else:
        blank_image = np.zeros((580, 580, 3), dtype=np.uint8)
        st.markdown("Press :green[▶️] to Detect Driver Drowsiness")
        cv2.putText(blank_image, "No Input", (250, 290), 1, 0.8, (255,255,255), 1)
        st.image(blank_image)

    
# Run the Streamlit app
if __name__ == "__main__":
    main()
    # Footer design
    config = json.load(open(".\\config.json"))
    st.markdown(
        f"""
        ___
        
        <style>
            footer {{
                color: #fff;
                text-align: center;
                padding: 1rem;
                position: relative;
                bottom: 0;
                width: 100%;
            }}
            a {{
                color: #fff;
                text-decoration: none;
                font-weight: bold;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
        <footer>
            <a href="{config["Linkedin"]}" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="{config["Linkedin"]}" height="20" width="30" /></a>
            <a href="{config["Github"]}" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg" height="20" width="30" /></a>
            <a href="https://mail.google.com/mail/?view=cm&fs=1&to=ziaul.karim497@gmail.com" target="_blank"><img align="center" src="https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_%282020%29.svg" height="15" width="30" /></a>
            <br>
            &copy; {dt.datetime.now().year} Made by - <a href= "https://ziaulkarim.netlify.app/" target="_blank"> Md. Ziaul Karim </a>
        </footer>
        """,
        unsafe_allow_html=True
    )