import random 
from utils.drawing import Draw
from utils.preprocess import Enhance
from utils.calculations import Calculate
from utils.custom_mediapipe import Points, Face, Hands

"""
The Driver Drowsiness class leverages on the functionality of the other classes such as 
- Enhance
- Calculate
- Points
- Face
- Hands
to detect Driver Drowsiness in Real Time.
"""
class DriverDrowsiness:
    
    def __init__(self, keypoints, frame_size=(480, 480),
                 mor_threshold = 0.6, ear_threshold = 0.25) -> None:
        
        # Custom utils package classes
        self.Enhance = Enhance() # used for enhancing image
        self.Calculate = Calculate() # used for calculations
        self.Points = Points() # used for custom facial and hand landmarks
        self.face_mesh = Face()
        self.hand = Hands()
        self.drawing = Draw() # used for drawings in the frame
        
        # aliasing the functions
        self.get_face_landmarks = self.face_mesh.process
        self.get_hand_landmarks = self.hand.process
        self.illuminate = self.Enhance.illumination_enhancement
        self.resize = self.Enhance.resize_image
        self.get_points = self.Points.get_points
        self.mid_point = self.Calculate.mid_point_finder
        self.intersect = self.Calculate.check_intersection
        self.EAR = self.Calculate.eye_aspect_ratio
        self.MOR = self.Calculate.mouth_open_ratio
        self.MID_MOR = self.Calculate.mid_mouth_open_ratio
        self.draw_eyes = self.drawing.draw_eye_features
        self.draw_mouth = self.drawing.draw_mouth_features
        self.draw_all = self.drawing.draw_all
        self.draw_intersect = self.drawing.draw_intersect
        
        # threshold for mouth and eye opening ratio
        self.mor_threshold = mor_threshold
        self.ear_threshold = ear_threshold
        self.last_mor = float('-inf')
        # custom frame size
        self.frame_size = frame_size
        
        # custom key points 
        self.LEFT_EYE = tuple(keypoints["left_eye_points"])
        self.RIGHT_EYE = tuple(keypoints["right_eye_points"])
        self.LIPS = tuple(keypoints["lips"])
        self.NOSE_TO_CHIN = tuple(keypoints["nose_to_chin"])
        self.HANDS = tuple(keypoints["hand_keypoints"])
        
    def EyeOpen(self, ear):
        return ear > self.ear_threshold
    
    def Yawning(self, mor):
        return mor < self.mor_threshold

  # Process the frame with facial key points
    def process_frame(self, frame):
        # Resize frame for faster processing (adjust as needed)
        frame = self.resize(frame, self.frame_size)
        
        # Convert to RGB for MediaPipe models
        rgb_frame = self.Enhance.bgr_to_rgb(frame)
        
        # adjust brightness & contrast

        process_frame = self.illuminate(rgb_frame)

            
        #get frame height, width and channel
        h, w = self.frame_size      
        
        # --------------------------- FACE LANDMARKS -----------------------------
        
        # Face Landmarks
        results_face = self.get_face_landmarks(process_frame)
        if results_face:
            #coordinates for inner right eye 
            
            right_eye_points = self.get_points(results_face, self.RIGHT_EYE, w, h)
            
            # coordinates for inner left eye
            left_eye_points = self.get_points(results_face, self.LEFT_EYE, w, h)
            
            # coordinates point for inner lips
            lips = self.get_points(results_face, self.LIPS, w, h)
            l1, l2, l3, l4 = lips
            
            # coordinates point for nose to chin
            nose_to_chin = self.get_points(results_face, self.NOSE_TO_CHIN, w, h)
            
            # all feature points
            feature_points = left_eye_points+right_eye_points+lips
            
            # individual EAR
            ear_left = self.EAR(args= left_eye_points)
            ear_right = self.EAR(args= right_eye_points)
            
            # overall EAR
            ear = round(min(ear_left, ear_right), 2)
            
            # mouth height mid point
            h_mid = self.mid_point(l2, l4)
            
            # mouth width mid point
            w_mid = self.mid_point(l1, l3)
            
            # approx center of the mouth
            center_mouth = self.mid_point(h_mid, w_mid)
            
            # mouth measures
            mor = round(max(self.MOR(args=lips), 
                        self.MID_MOR(h_mid, w_mid, lips)),
                        2)
            eye_open = self.EyeOpen(ear)
            yawning = self.Yawning(mor)
            self.last_mor = mor if mor>self.mor_threshold else self.last_mor
            
            # --------------------------- HANDS LANDMARKS -----------------------------

            results_hands = self.get_hand_landmarks(process_frame)
            if results_hands:
                hand_points = self.get_points(results_hands, self.HANDS, w, h)
                index_finger_mcp, finger_tips = hand_points[0], hand_points[1:]
                try:
                    for finger_tip in finger_tips:
                        if not eye_open \
                        and (
                            self.intersect(nose_to_chin,
                                            (index_finger_mcp,
                                                finger_tip))
                            ):    
                            mor = round(random.uniform(self.mor_threshold, self.last_mor),2)
                            self.draw_intersect(index_finger_mcp, finger_tip, frame, nose_to_chin)

                            break
                        else:
                            continue
                    
                except Exception as e:
                    print(f"Error: {e}")
            
            # -------------------------------- DRAWING DETECTED FEATURES -----------------------------------
            
            # drawing the eye features        
            self.draw_eyes(eye_open, ear, frame, left_eye_points, right_eye_points)
            
            #drawing mouth features
            self.draw_mouth(yawning, mor, frame, lips)

            # draw all features
            self.draw_all(frame, feature_points, center_mouth)
                    
        return frame