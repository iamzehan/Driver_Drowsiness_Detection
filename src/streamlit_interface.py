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
            duration_threshold = 4
            mor_threshold = 0.6
            ear_threshold = 0.25
            start_time = time.time()
            sleep_count = 0
            detector = DriverDrowsiness(keypoints=keypoints)
            draw = Draw()
            with col2: stop = st.button(":red[🟥]")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if stop:
                    # Release the captured frame
                    cap.release()
                    break
                # Processing frame
                results = detector.process_frame(frame)
                
                try:
                    result_frame, mor, ear, head_point = results
                    # Alerting based on elapsed sleep time
                    # ear_count = st.text(f"EAR: `{ear_count}`")
                    is_sleepy = (mor > mor_threshold or ear < ear_threshold)
                    if is_sleepy:
                        # Calculate elapsed time sleeping and alert the driver
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        sleep_count += 1
                        if elapsed_time >= duration_threshold:
                            draw.draw_sleepy(result_frame, head_point)
                            play_alarm()

                    else:
                        # Reset the counters
                        start_time = time.time()
                        sleep_count = 0
                        
                    # Display the processed frame in Streamlit
                    video_placeholder.image(result_frame, channels="BGR", use_column_width=True, output_format="JPEG")
                    
                    
                except:
                    result_frame = results
                    # Reset the counters
                    start_time = time.time()
                    sleep_count = 0
                    
                # FPS count
                cTime = time.time()
                fps = fps_count(cTime, pTime)
                pTime = cTime
                draw.draw_fps_count(result_frame, fps)
                # Display the processed frame in Streamlit
                video_placeholder.image(result_frame, channels="BGR", use_column_width=True, output_format="JPEG")

        else:
            st.error("Error: Could not open video")
    else:
        st.markdown("Press :green[▶️] to Detect Driver Drowsiness")
# Run the Streamlit app
if __name__ == "__main__":
    main()
