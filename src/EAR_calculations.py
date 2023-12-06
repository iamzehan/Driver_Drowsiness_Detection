import dlib
import json

# Read the configuration file
with open('.\config.json', 'r') as config_file:
    config_data = json.load(config_file)

# Access the MongoDB URI
PATH = config_data['IMG_PATH']
# Function to calculate Eye Aspect Ratio (EAR) based on facial landmarks
def calculate_ear_from_landmarks(landmarks):
    # Extracting coordinates of the eyes
    left_eye = [landmarks.part(i).x for i in range(36, 42)]  # Indices 36 to 41 correspond to the left eye
    right_eye = [landmarks.part(i).x for i in range(42, 48)]  # Indices 42 to 47 correspond to the right eye

    # Calculating EAR
    left_ear = (abs(left_eye[1] - left_eye[5]) + abs(left_eye[2] - left_eye[4])) / (2.0 * abs(left_eye[0] - left_eye[3]))
    right_ear = (abs(right_eye[1] - right_eye[5]) + abs(right_eye[2] - right_eye[4])) / (2.0 * abs(right_eye[0] - right_eye[3]))

    # Overall EAR is the average of left and right EAR
    ear = (left_ear + right_ear) / 2.0

    return ear

# Example usage with facial landmarks from dlib
# Assuming you have a face detector and shape predictor initialized
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models\shape_predictor_68_face_landmarks.dat")  # Replace with the actual path

# Example image (replace with your image)
image = dlib.load_rgb_image(F"{PATH}")  # Replace with the actual path

# Detect faces in the image
faces = detector(image)

# Assuming there's at least one face detected
landmarks = predictor(image, faces[0])

# Calculate EAR using the detected landmarks
ear_value = calculate_ear_from_landmarks(landmarks)

print("Eye Aspect Ratio:", ear_value)
