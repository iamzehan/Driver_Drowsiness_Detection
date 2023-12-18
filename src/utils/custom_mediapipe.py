import mediapipe as mp

class CustomMediaPipe:
    def __init__(self, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
                mor_threshold = 0.6, ear_threshold = 0.25) -> None:
    
        
        # Mediapipe face mesh functionalities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            
        )
        
        # threshold for mouth and eye opening ratio
        self.mor_threshold = mor_threshold
        self.ear_threshold = ear_threshold
        
        # Mediapipe hands functionalities
        self.mp_hands = mp.solutions.hands
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence= min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1 # Adjusted tracking confidence

        )
    
class Hands(CustomMediaPipe):
    def process(self,process_frame):
        return self.hands.process(process_frame).multi_hand_landmarks
    
class Face(CustomMediaPipe):
    def process(self,process_frame):
        return self.face_mesh.process(process_frame).multi_face_landmarks

class Points:    
    def get_points(self, results, parts, w, h):
        
            # find the actual coordinates of the point
        def point_finder(face_landmarks, point, w, h):
            return (int(face_landmarks.landmark[point].x*w),
                    int(face_landmarks.landmark[point].y*h))
        points = [
                point_finder(landmarks, kp, w, h) 
                    for landmarks in results 
                        for kp in parts
                ]
        return points
    