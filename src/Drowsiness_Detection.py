import cv2
import json
import time
import numpy as np
import mediapipe as mp
from utils.Image_Enhance import Enhance
from utils.Caclulations import Calculate
from utils.Custom_Keypoints import KeyPoints
from utils.draw import Draw

class DriverDrowsiness:
    
    def __init__(self, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 mor_threshold = 0.6, ear_threshold = 0.25) -> None:
        
        # Custom utils package
        self.Enhance = Enhance() # used for enhancing image
        self.Calculate = Calculate() # used for calculations
        self.KeyPoints = KeyPoints() # used for cutom facial and hand landmarks
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

  # Process the frame with facial key points
    def process_frame(self, frame: np.array, frame_size = (480, 480)) -> np.array:
        
        # Resize frame for faster processing (adjust as needed)
        frame = cv2.resize(frame, frame_size)
        
        # Convert to RGB for MediaPipe models
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # adjust brightness & contrast
        try:
            process_frame = self.Enhance.illumination_enhancement(rgb_frame)
        except Exception as e:
            print(f"Error: {e}")
            
        #get frame height, width and channel
        h, w, c = frame.shape        
        
        # --------------------------- FACE LANDMARKS -----------------------------
        # Face Landmarks
        results_face = self.face_mesh.process(process_frame)
        if results_face.multi_face_landmarks:
            #coordinates for inner right eye 
            right_eye_points = [
                self.Calculate.point_finder(face_landmarks, kp, w, h) 
                    for face_landmarks in results_face.multi_face_landmarks 
                        for kp in self.KeyPoints.RIGHT_EYE
                ]
            
            
            # coordinates for inner left eye
            left_eye_points = [
                self.Calculate.point_finder(face_landmarks, kp, w, h) 
                    for face_landmarks in results_face.multi_face_landmarks 
                        for kp in self.KeyPoints.LEFT_EYE
                ]
            
            
            # coordinates point for inner lips
            lips = [
                self.Calculate.point_finder(face_landmarks, kp, w, h) 
                    for face_landmarks in results_face.multi_face_landmarks 
                        for kp in self.KeyPoints.LIPS
                    ]
            l1, l2, l3, l4 = lips
            
            nose_to_chin = [
                self.Calculate.point_finder(face_landmarks, kp, w, h) 
                    for face_landmarks in results_face.multi_face_landmarks 
                        for kp in self.KeyPoints.NOSE_TO_CHIN
                ]

            # all feature points
            feature_points = left_eye_points+right_eye_points+lips
                    
            # individual EAR
            ear_left = self.Calculate.calculate_ear_from_landmarks(args= left_eye_points)
            ear_right = self.Calculate.calculate_ear_from_landmarks(args= right_eye_points)
            
            # overall EAR
            ear = round(min(ear_left, ear_right), 2)
                    
            frame = self.Draw.draw_eye_property(ear, frame, left_eye_points, right_eye_points)
            
            # mouth height mid point
            h_mid = self.Calculate.mid_point_finder(l2, l4)
            # mouth width mid point
            w_mid = self.Calculate.mid_point_finder(l1, l3)
            
            # mouth measures
            mor = round(max(self.Calculate.mouth_open_ratio(args=lips), 
                        self.Calculate.mid_mouth_open_ratio(h_mid, w_mid, lips)), 2)
            
            #drawing mouth features
            frame = self.Draw.draw_mouth_property(mor, frame, lips)
            
             # drawing all the features
            for point in feature_points:
                cv2.circle(frame,
                            center = point,
                            radius= 1,
                            color = (0, 255, 0),
                            thickness=-1)
                
            # drawing the approximate mid point
            cv2.circle(frame,
                    center = self.Calculate.mid_point_finder(h_mid, w_mid), 
                    radius= 1, 
                    color = (0, 255, 0),
                    thickness=1)
            
            # --------------------------- HANDS LANDMARKS -----------------------------

            results_hands = self.hands.process(rgb_frame)
            if results_hands.multi_hand_landmarks:
                hand_points = [
                        self.Calculate.point_finder(hand_landmarks, kp, w, h) 
                            for hand_landmarks in results_hands.multi_hand_landmarks 
                                for kp in self.KeyPoints.HANDS
                        ]
                
                try:
                    for i in range(1,len(hand_points)):
                        if (ear < self.ear_threshold ) \
                        and (
                            self.Calculate.check_intersection(nose_to_chin,
                                                    [hand_points[0],
                                                    hand_points[i]])
                            ):    
                            
                            frame = self.Draw.draw_intersect(i, frame, nose_to_chin, hand_points)

                            break
                        else:
                            continue
                    
                except Exception as e:
                    print(f"Error: {e}")
        return frame

if __name__ == "__main__":
    
    # reading the config file to get camera path
    config_data = json.load(open("config.json", "r"))
    
    # getting the path from the file
    VIDEO_PATH = config_data['IP_CAM']["phone"]
    
    cap = cv2.VideoCapture("Videos/3.mp4")

    #feeding the frames in a loop
    if cap.isOpened():
        pTime = 0
        detector = DriverDrowsiness()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    
            result_frame = detector.process_frame(frame)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.startWindowThread()
            
            # for visibility of the FPS count
            cv2.rectangle(result_frame,
                          (15, 60),
                          (70, 72),
                          color=(0, 0, 0),
                          thickness=-1)
            
            # Writing the fps count 
            cv2.putText(result_frame,
                        f'FPS:{int(fps)}',
                        (20, 70),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.8,
                        (255, 255, 255),
                        1)
            cv2.imshow("Output", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # if the video doesn't open
    else:
        print("Error: Could not open video")
    
    # release the captured frame
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Needed to close the imshow window