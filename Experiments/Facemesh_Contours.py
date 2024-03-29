import cv2
import json
import mediapipe as mp


class FaceMeshDetector:
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

    def process_image(self, image):
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return

        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

        cv2.imshow("Output", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        
if __name__ == "__main__":
    with open('.\config.json', 'r') as config_file:
        config_data = json.load(config_file)
    PATH = config_data['IMG_PATH']
    img = cv2.imread(PATH)
    detector = FaceMeshDetector()
    detector.process_image(img)