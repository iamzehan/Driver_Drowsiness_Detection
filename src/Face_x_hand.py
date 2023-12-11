import cv2 
import json
import numpy as np 
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
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5)
        
    def process_image(self, image):
        
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # there are no landmarks in the image, then we return the function
        if not results.multi_face_landmarks:
            return
        
        # we don't want to use the original image for drawing
        annotated_image = image.copy()
        
        # draw face landmarks 
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

        # hand LandMarks
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
        
        return annotated_image
        
if __name__ == "__main__":
    PATH = "Attatchments\A-tired-man-yawning-behind-the-wheel-of-his-car.png"
    img = cv2.imread(PATH)
    w, h, c = img.shape
    detector = FacexHandDetector()
    result = detector.process_image(img)
    result = cv2.resize(result, (h//2, w//2))
    cv2.startWindowThread()
    cv2.imshow("Output", result)
    cv2.imwrite('Image Outputs/Output_A-tired-man-yawning-behind-the-wheel-of-his-car.png', result)
    cv2.waitKey(0)