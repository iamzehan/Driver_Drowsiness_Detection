import cv2
import json
import dlib
import math

# Load pre-trained face and landmark detectors from dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Capture video from the camera
config_data = json.load(open('config.json', 'r'))

# I am using IP_CAM android app, 
# so I stored my IP address in a config file,
# for security reasons.
cap = cv2.VideoCapture(config_data['IP_CAM']['tab'])

# Nose points in the 68-point landmark model
nose_points = [31, 35]

# Average nose length while awake (adjust this based on your observations)
average_nose_length_awake = 20.0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (480, 480))
    if not ret:
        break
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    for face in faces:
        # Detect landmarks in the face region
        landmarks = landmark_predictor(gray, face)

        # Extract nose points
        nose = [(landmarks.part(point).x, landmarks.part(point).y) for point in nose_points]

        # Calculate nose length
        nose_length = math.sqrt((nose[1][0] - nose[0][0])**2 + (nose[1][1] - nose[0][1])**2)

        # Calculate the ratio of nose length to average nose length
        nose_length_ratio = nose_length / average_nose_length_awake

        # Display the nose length and ratio on the frame
        cv2.putText(frame, f"Nose Length: {nose_length:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Nose Length Ratio: {nose_length_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Check for drowsiness (adjust the threshold as needed)
        drowsy_threshold = 1.5
        if nose_length_ratio > drowsy_threshold or nose_length_ratio < 1 / drowsy_threshold:
            cv2.putText(frame, "Drowsiness Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
