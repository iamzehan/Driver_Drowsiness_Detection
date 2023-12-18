import mediapipe as mp

"""
Hands class contains the mediapipe Hands landmark model
process() function in the Hand class returns the landmarks for the detectied hands

Face class contains the mediapipe FaceMesh landmark model
process() function in the Face class returns the landmarks for the detected face

Points class contains the get_points() function 
which returns the (x, y) coordinates of multiple predefined landmarks

"""
class Hands:
    def __init__(self,
                 max_num_hands = 1,
                 model_complexity = 1, 
                 min_detection_confidence = 0.5,
                 min_tracking_confidence=0.5) -> None:
    
    # Mediapipe hands functionalities
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence= min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity # Adjusted tracking confidence

        )
    def process(self,process_frame):
        return self.hands.process(process_frame).multi_hand_landmarks
    
class Face:
    def __init__(self, 
                 max_num_faces=1, 
                 refine_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5
                ) -> None:
        
        # Mediapipe face mesh functionalities
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
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
        return tuple(points)
    