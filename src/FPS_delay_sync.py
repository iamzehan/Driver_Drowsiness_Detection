import cv2
import time

# Open the video file or capture device
video_path = 'Videos/3.mp4'  # Replace with your video file path or capture device index
cap = cv2.VideoCapture(video_path)

# Get the original FPS of the video
original_fps = cap.get(cv2.CAP_PROP_FPS)
print(original_fps)
# Check if the video capture is successful
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
dl=[]
while True:
    # Start time for measuring processing time
    start_time = time.time()

    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the video has ended
    if not ret:
        print("Video has ended.")
        break

    # Your processing code goes here

    # Display the frame
    cv2.imshow('Video', frame)

    # Calculate the time taken to process the frame
    processing_time = time.time() - start_time

    # Calculate the delay required to achieve the desired FPS
    delay = max(1, int((1 / original_fps - processing_time) * 750))
    dl.append(delay)
    # If processing is too fast, add a small additional delay
    if delay < 1:
        time.sleep(0.001)

    # Wait for the specified delay or until a key is pressed
    key = cv2.waitKey(delay) & 0xFF

    # Break the loop if the 'q' key is pressed
    if key == ord('q') or key==(27):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
