import cv2
import json
import mediapipe as mp

class FacexHandDetector:
    def __init__(self, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_hands = mp.solutions.hands
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3  # Adjusted tracking confidence
        )

    def process_frame(self, frame):
        # Resize frame for faster processing (adjust as needed)
        #frame = cv2.resize(frame, (640, 480))

        # Convert to RGB for MediaPipe models
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face Landmarks
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                feature_set = [self.mp_face_mesh.FACEMESH_LEFT_EYE, self.mp_face_mesh.FACEMESH_RIGHT_EYE, self.mp_face_mesh.FACEMESH_LIPS]
                for feature in feature_set:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=feature,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(frame, center = (x, y), radius= 1, color = (0, 0, 255))
        # Hand LandMarks
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        return frame

if __name__ == "__main__":
    with open("config.json", "r") as config_file:
        config_data = json.load(config_file)
    VIDEO_PATH = config_data['IP_CAM']
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    detector = FacexHandDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = detector.process_frame(frame)

        cv2.imshow("Output", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Needed to close the imshow window
