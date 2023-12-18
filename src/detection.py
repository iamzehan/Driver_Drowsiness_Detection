from utils.draw import Draw
from utils.preprocess import Enhance
from utils.calculations import Calculate
from utils.custom_mediapipe import Points, Face, Hands


class DriverDrowsiness:
    
    def __init__(self, keypoints, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 mor_threshold = 0.6, ear_threshold = 0.25) -> None:
        
        # Custom utils package
        self.Enhance = Enhance() # used for enhancing image
        self.Calculate = Calculate() # used for calculations
        self.Points = Points() # used for custom facial and hand landmarks
        self.face_detect = Face()
        self.hand_detect = Hands()
        self.Draw = Draw() # used for drawings in the frame
        
        # threshold for mouth and eye opening ratio
        self.mor_threshold = mor_threshold
        self.ear_threshold = ear_threshold
        
        # custom key points 
        self.LEFT_EYE = keypoints["left_eye_points"]
        self.RIGHT_EYE = keypoints["right_eye_points"]
        self.LIPS = keypoints["lips"]
        self.NOSE_TO_CHIN = keypoints["nose_to_chin"]
        self.HANDS = keypoints["hand_keypoints"]
        
    def EyeFullyOpen(self, ear):
        return ear > self.ear_threshold
    
    def isYawning(self, mor):
        return mor > self.mor_threshold

  # Process the frame with facial key points
    def process_frame(self, frame, frame_size = (480, 480)):
        # Resize frame for faster processing (adjust as needed)
        frame = self.Enhance.resize_image(frame, frame_size)
        
        # Convert to RGB for MediaPipe models
        rgb_frame = self.Enhance.bgr_to_rgb(frame)
        
        # adjust brightness & contrast

        process_frame = self.Enhance.illumination_enhancement(rgb_frame)

            
        #get frame height, width and channel
        h, w = frame_size      
        
        # --------------------------- FACE LANDMARKS -----------------------------
        # Face Landmarks
        try:
            results_face = self.face_detect.process(process_frame)
        except Exception as e:
            print(e)
        
        if results_face:
            #coordinates for inner right eye 
            right_eye_points = self.Points.get_points(results_face, self.RIGHT_EYE, w, h)

            # coordinates for inner left eye
            left_eye_points = self.Points.get_points(results_face, self.LEFT_EYE, w, h)
            
            # coordinates point for inner lips
            lips = self.Points.get_points(results_face, self.LIPS, w, h)
            l1, l2, l3, l4 = lips
            
            nose_to_chin = self.Points.get_points(results_face, self.NOSE_TO_CHIN, w, h)

            # all feature points
            feature_points = left_eye_points+right_eye_points+lips

            # individual EAR
            ear_left = self.Calculate.calculate_ear_from_landmarks(args= left_eye_points)
            ear_right = self.Calculate.calculate_ear_from_landmarks(args= right_eye_points)
            
            # overall EAR
            ear = round(min(ear_left, ear_right), 2)
            
            # drawing the eye features        
            self.Draw.draw_eye_features(self.EyeFullyOpen(ear), ear, frame, left_eye_points, right_eye_points)
            
            # mouth height mid point
            h_mid = self.Calculate.mid_point_finder(l2, l4)
            
            # mouth width mid point
            w_mid = self.Calculate.mid_point_finder(l1, l3)
            
            # approx center of the mouth
            center_mouth = self.Calculate.mid_point_finder(h_mid, w_mid)
            
            # mouth measures
            mor = round(max(self.Calculate.mouth_open_ratio(args=lips), 
                        self.Calculate.mid_mouth_open_ratio(h_mid, w_mid, lips)),
                        2)
            
            #drawing mouth features
            self.Draw.draw_mouth_features(self.isYawning(mor), mor, frame, lips)

            # draw all features
            self.Draw.draw_all(frame, feature_points, center_mouth)
            
            # --------------------------- HANDS LANDMARKS -----------------------------

            results_hands = self.hand_detect.process(process_frame)
            if results_hands:
                hand_points = self.Points.get_points(results_hands, self.HANDS, w, h)
                
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