import cv2
import numpy as np
import mediapipe as mp
from utils.Image_Enhance import Enhance
from utils.Calculations import Calculate
from utils.FaceMesh_Custom_Keypoints import KeyPoints
from utils.draw import Draw

class DriverDrowsiness:
    
    def __init__(self, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 mor_threshold = 0.6, ear_threshold = 0.25) -> None:
        
        # Custom utils package
        self.Enhance = Enhance() # used for enhancing image
        self.Calculate = Calculate() # used for calculations
        self.KeyPoints = KeyPoints() # used for custom facial and hand landmarks
        self.Draw = Draw() # used for drawings in the frame
        
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
    def EyeFullyOpen(self, ear):
        return ear>self.ear_threshold
    

  # Process the frame with facial key points
    def process_frame(self, frame: np.array, frame_size = (480, 480)) -> np.array:
        
        # Resize frame for faster processing (adjust as needed)
        frame = cv2.resize(frame, frame_size)
        
        # Convert to RGB for MediaPipe models
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # adjust brightness & contrast

        process_frame = self.Enhance.illumination_enhancement(rgb_frame)

            
        #get frame height, width and channel
        h, w = frame_size      
        
        # --------------------------- FACE LANDMARKS -----------------------------
        # Face Landmarks
        results_face = self.face_mesh.process(process_frame).multi_face_landmarks
        if results_face:
            #coordinates for inner right eye 
            
            right_eye_points = self.KeyPoints.get_points(results_face, self.KeyPoints.RIGHT_EYE, w, h)
            
            # coordinates for inner left eye
            left_eye_points = self.KeyPoints.get_points(results_face, self.KeyPoints.LEFT_EYE, w, h)
            
            # coordinates point for inner lips
            lips = self.KeyPoints.get_points(results_face, self.KeyPoints.LIPS, w, h)
            l1, l2, l3, l4 = lips
            
            nose_to_chin = self.KeyPoints.get_points(results_face, self.KeyPoints.NOSE_TO_CHIN, w, h)

            # all feature points
            feature_points = left_eye_points+right_eye_points+lips
                    
            # individual EAR
            ear_left = self.Calculate.calculate_ear_from_landmarks(args= left_eye_points)
            ear_right = self.Calculate.calculate_ear_from_landmarks(args= right_eye_points)
            
            # overall EAR
            ear = round(min(ear_left, ear_right), 2)
            
            # drawing the eye features        
            self.Draw.draw_eye_features(ear, frame, left_eye_points, right_eye_points)
            
            # mouth height mid point
            h_mid = self.Calculate.mid_point_finder(l2, l4)
            
            # mouth width mid point
            w_mid = self.Calculate.mid_point_finder(l1, l3)
            
            # approx center of the mouth
            center_mouth = self.Calculate.mid_point_finder(h_mid, w_mid)
            
            # mouth measures
            mor = round(max(self.Calculate.mouth_open_ratio(args=lips), 
                        self.Calculate.mid_mouth_open_ratio(h_mid, w_mid, lips)), 2)
            
            #drawing mouth features
            self.Draw.draw_mouth_features(mor, frame, lips)

            
            
            # draw all features
            self.Draw.draw_all(frame, feature_points, center_mouth)
            
            # --------------------------- HANDS LANDMARKS -----------------------------

            results_hands = self.hands.process(rgb_frame).multi_hand_landmarks
            if results_hands:
                hand_points = self.KeyPoints.get_points(results_hands, self.KeyPoints.HANDS, w, h)
                
                try:
                    for i in range(1,len(hand_points)):
                        if not self.EyeFullyOpen(ear) \
                        and (
                            self.Calculate.check_intersection(nose_to_chin,
                                                    [hand_points[0],
                                                    hand_points[i]])
                            ):    
                            
                            self.Draw.draw_intersect(i, frame, nose_to_chin, hand_points)

                            break
                        else:
                            continue
                    
                except Exception as e:
                    print(f"Error: {e}")
                    
        return frame