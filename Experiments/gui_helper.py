import cv2
import json
import dlib
import math
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def calculate_angle(point1, point2, point3):
    def vector_length(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    # Calculate vectors
    vector1 = (point2[0] - point1[0], point2[1] - point1[1])
    vector2 = (point3[0] - point1[0], point3[1] - point1[1])

    # Calculate dot product
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate vector lengths
    length1 = vector_length(point1, point2)
    length2 = vector_length(point1, point3)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (length1 * length2)

    # Calculate the angle in radians
    angle_radians = math.acos(cos_angle)

    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

def start_drowsiness_detection(source):
    # Load pre-trained face and landmark detectors from dlib
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Nose points in the 68-point landmark model
    nose_points = [27, 30]

    # Average nose length while awake (adjust this based on your observations)
    average_nose_length_awake = 20.0

    # Capture video from the selected source
    config_data = json.load(open("config.json", "r"))
    addr = config_data['IP_CAM'][source]
    cap = cv2.VideoCapture(addr)
    
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (480, 480))
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector(gray)
        h, w, c = frame.shape
        for face in faces:
            # Detect landmarks in the face region
            landmarks = landmark_predictor(gray, face)

            # Extract nose points
            nose = [(landmarks.part(point).x, landmarks.part(point).y) for point in nose_points]
            cv2.line(frame, nose[0], nose[1],color=(255,0,0), thickness=1)
            # Calculate nose length
            nose_length = math.sqrt((nose[1][0] - nose[0][0])**2 + (nose[1][1] - nose[0][1])**2)
            cv2.line(frame, (0, h//2), (nose[1]), color=(255,0,0), thickness=1)
            cv2.line(frame, (0, h//2), (nose[0]), color=(0,0,255), thickness=1)
            
            # Calculate angle between lines
            
            angle_degrees = calculate_angle((0, h//2), nose[1], nose[0])

            # Display the angle on the frame
            cv2.putText(frame, f"Angle: {angle_degrees:.2f} degrees", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            # Calculate the ratio of nose length to average nose length
            nose_length_ratio = nose_length / average_nose_length_awake
            drowsy_threshold = 1.5
            # Display the nose length and ratio on the frame
            cv2.putText(frame, f"Nose Length: {nose_length:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Nose Length Ratio: {nose_length_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Check for drowsiness (adjust the threshold as needed)
            
            if nose_length_ratio > drowsy_threshold or nose_length_ratio < 1 / drowsy_threshold:
                cv2.putText(frame, "Drowsiness Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame in the Tkinter window
        cv2.startWindowThread()
        cv2.imshow("Detection", frame)
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    exit()

# Create the main Tkinter window
root = tk.Tk()
root.title("Drowsiness Detection")
root.geometry("640x480")
root.title("Drowsiness Detection")

# Create a dropdown for choosing the video source
source_label = ttk.Label(root, text="Choose Video Source:")
source_label.pack(pady=10)

source_var = tk.StringVar()
source_combobox = ttk.Combobox(root, textvariable=source_var, values=["tab", "phone"])
source_combobox.pack(pady=10)
source_combobox.set("phone")  # Default to webcam (change this if needed)

start_button = ttk.Button(root, text="Start Detection", command=lambda: start_drowsiness_detection(source_var.get()))
start_button.pack(pady=20)

# Start the Tkinter main loop
root.mainloop()
