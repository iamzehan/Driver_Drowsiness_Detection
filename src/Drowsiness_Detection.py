import cv2
import json
import time
import mediapipe as mp
from utils.Image_Enhance import Enhance
from utils.Caclulations import Calculate

class DriverDrowsiness:
    
    def __init__(self, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # face mesh functionalities
        self.Enhance = Enhance()
        self.Calculate = Calculate()
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
        self.mor_threshold = 0.6
        self.ear_threshold = 0.25
        
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
    def process_frame(self, frame, key_points):
        # Resize frame for faster processing (adjust as needed)
        
        frame = cv2.resize(frame, (480, 480))
        
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
                        for kp in key_points["right_eye_points"]
                ]
            rp1, rp2, rp3, rp4, rp5, rp6 = right_eye_points
            
            # coordinates for inner left eye
            left_eye_points = [
                self.Calculate.point_finder(face_landmarks, kp, w, h) 
                    for face_landmarks in results_face.multi_face_landmarks 
                        for kp in key_points["left_eye_points"]
                ]
            lp1, lp2, lp3, lp4, lp5, lp6 = left_eye_points
            
            # coordinates point for inner lips
            lips = [
                self.Calculate.point_finder(face_landmarks, kp, w, h) 
                    for face_landmarks in results_face.multi_face_landmarks 
                        for kp in key_points["lips"]
                    ]
            l1, l2, l3, l4 = lips
            
            nose_to_chin = [
                self.Calculate.point_finder(face_landmarks, kp, w, h) 
                    for face_landmarks in results_face.multi_face_landmarks 
                        for kp in key_points["nose_to_chin"]
                ]
            nch1, nch2 = nose_to_chin
            
            # all feature points
            feature_points = left_eye_points+right_eye_points+lips
                    
            # individual EAR
            ear_left = self.Calculate.calculate_ear_from_landmarks(args= left_eye_points)
            ear_right = self.Calculate.calculate_ear_from_landmarks(args= right_eye_points)
            
            # overall EAR
            ear = round(min(ear_left, ear_right), 2)
                    
            #For better visibility of EAR and MOR text
            cv2.rectangle(frame,
                            pt1=(230, 410),
                            pt2=(320, 460),
                            color=(0,0,0),
                            thickness= -1)
            
            # left eye width line
            cv2.line(frame,
                        pt1=lp1,
                        pt2=lp4,
                        color=(255,255,255) 
                        if ear > self.ear_threshold 
                        else (0, 0, 255), 
                        thickness = 1)
            
            # right eye width line
            cv2.line(frame,
                        pt1=rp1,
                        pt2=rp4,
                        color=(255,255,255) 
                        if ear > self.ear_threshold 
                        else (0, 0, 255),
                        thickness = 1)
            
            # writing EAR on the screen 
            cv2.putText(frame,
                        f"EAR: {ear}", 
                        (240, 430),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.8,
                        color = (0,255,0) 
                        if ear > self.ear_threshold 
                        else (0, 0, 255),
                        thickness=1)
            
            
            # mouth height mid point
            h_mid = self.Calculate.mid_point_finder(l2, l4)
            
            # mouth width mid point
            w_mid = self.Calculate.mid_point_finder(l1, l3)
            
            # mouth measures
            mor = round(max(self.Calculate.mouth_open_ratio(args=lips), 
                        self.Calculate.mid_mouth_open_ratio(h_mid, w_mid, lips)), 2)
            
            # mouth width line
            cv2.line(frame, 
                        pt1=l1,
                        pt2=l3,
                        color=(255,255,255) 
                        if mor < self.mor_threshold 
                        else (0, 0, 255),
                        thickness = 1)
            
            # mouth height line 
            cv2.line(frame, 
                        pt1=l2,
                        pt2=l4,
                        color=(255,255,255) 
                        if mor < self.mor_threshold 
                        else (0, 0, 255),
                        thickness = 1)
            
            # writing the mor on the screen
            cv2.putText(frame,
                        f"MOR: {mor}",
                        (240, 450), 
                        cv2.FONT_HERSHEY_PLAIN, 
                        0.8, 
                        color = (0,255,0) 
                        if mor < self.mor_threshold 
                        else (0, 0, 255),
                        thickness=1)
            
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
                                for kp in key_points["hand_keypoints"]
                        ]
                
                try:
                    for i in range(1,len(hand_points)):
                        if (ear < self.ear_threshold ) \
                        and (
                            self.Calculate.check_intersection(nose_to_chin,
                                                    [hand_points[0],
                                                    hand_points[i]])
                            ):    
                            
                            
                            # drawing the intersecting lines
                            cv2.line(frame,
                                pt1=nch1,
                                pt2=nch2,
                                color=(0,0,255),
                                thickness = 1)
                            
                            cv2.line(frame, 
                                    pt1=hand_points[0],
                                    pt2=hand_points[i],
                                    color=(0,0,255),
                                    thickness = 1)
                            
                            # index knuckle
                            cv2.circle(frame,
                            center= hand_points[0],
                            radius=2,
                            color=(0,255,0),
                            thickness=-1)
                            
                            # any finger tip that intersects
                            cv2.circle(frame,
                            center= hand_points[i],
                            radius=2,
                            color=(0,255,0),
                            thickness=-1)
                            
                            cv2.putText(frame,
                            f"Status: Yawning!!",
                            (160, 400), 
                            cv2.FONT_HERSHEY_PLAIN, 
                            2, 
                            color = (0,0,255),
                            thickness=2)
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
    
    # getting facial key points, tailored to our needs
    key_points = json.load(open("src/config/config.json"))
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    #feeding the frames in a loop
    if cap.isOpened():
        pTime = 0
        detector = DriverDrowsiness()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    
            result_frame = detector.process_frame(frame, key_points)
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