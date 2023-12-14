import cv2
import json
import math
import time
import numpy as np
import mediapipe as mp

class FacexHandDetector:
    def __init__(self, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # threshold for mouth and eye opening ratio
        self.mor_threshold = 0.6
        self.ear_threshold = 0.25

    # find the actual coordinates of the point
    def point_finder(self, face_landmarks, point, w, h):
        return (int(face_landmarks.landmark[point].x*w), int(face_landmarks.landmark[point].y*h))
    
    # mid point finder
    def mid_point_finder(self, point1, point2, w, h, face_landmarks = None):
        if face_landmarks:
            x1, y1 = self.point_finder(face_landmarks, point1, w, h) 
            x2, y2 = self.point_finder(face_landmarks, point2, w, h) 
        else:
            x1, y1 = point1
            x2, y2 = point2
        return (x2 + x1)//2, (y2+y1)//2
    
    # calculate length between two points
    def length_calculation(self, face_landmarks, point1, point2, w, h):
        x_1, y_1 = self.point_finder(face_landmarks, point1, w, h)
        x_2, y_2 = self.point_finder(face_landmarks, point2, w, h)
        return math.sqrt((x_1-x_2)**2 + (y_1 - y_2)**2)

    # Function to calculate Eye Aspect Ratio (EAR) based on facial landmarks
    def calculate_ear_from_landmarks(self, face_landmarks, w, h, args):
        
        # Extracting coordinates of the eyes
        p1, p2, p3, p4, p5, p6 = args
        
        # Calculating Lengths
        A = self.length_calculation(face_landmarks, p2, p6, w, h)
        B = self.length_calculation(face_landmarks, p3, p5, w, h)
        C = self.length_calculation(face_landmarks, p1, p4, w, h)
        # calculating EAR
        ear = (A+B) / (2.0*C) 
        return ear

    # Mouth Open Ratio (MOR)
    def mouth_open_ratio(self, face_landmarks, w, h, args):
        l1,l2,l3,l4 = args 
        mouth_w, mouth_h = self.length_calculation(face_landmarks, l1, l3, w, h), self.length_calculation(face_landmarks, l2, l4, w, h)
        return round(mouth_h/mouth_w, 2)
    
    # Process the frame with facial key points
    def process_frame(self, frame, key_points):
        # Resize frame for faster processing (adjust as needed)
        frame = cv2.resize(frame, (480, 480))
        
        # Convert to RGB for MediaPipe models
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #get frame height, width and channel
        h, w, c = frame.shape        
        
        # Face Landmarks
        results_face = self.face_mesh.process(rgb_frame)
        
        #coordinates for inner right eye 
        right_eye_points = key_points["right_eye_points"]
        rp1, rp2, rp3, rp4, rp5, rp6 = right_eye_points
        
        # coordinates for inner left eye
        left_eye_points = key_points["left_eye_points"]
        lp1, lp2, lp3, lp4, lp5, lp6 = left_eye_points
        
        # coordinates point for inner lips
        lips = key_points["lips"]
        l1, l2, l3, l4 = lips
        
        # all feature points
        feature_points = left_eye_points+right_eye_points+lips
        
        if results_face.multi_face_landmarks:
            
            for face_landmarks in results_face.multi_face_landmarks:
                
                # individual EAR
                ear_left = self.calculate_ear_from_landmarks(face_landmarks, w, h, args= left_eye_points)
                ear_right = self.calculate_ear_from_landmarks(face_landmarks, w, h, args= right_eye_points)
                
                # overall EAR
                ear = round((ear_left+ear_right)/2.0, 2)
                
                #For better visibility of EAR and MOR text
                cv2.rectangle(frame,
                              pt1=(230, 430),
                              pt2=(330, 480),
                              color=(0,0,0),
                              thickness= -1)
                
                # left eye width line
                cv2.line(frame,
                         pt1=self.point_finder(face_landmarks, lp1, w, h), 
                         pt2=self.point_finder(face_landmarks, lp4, w, h),
                         color=(255,255,255) if ear > self.ear_threshold else (0, 0, 255), 
                         thickness = 1)
                
                # right eye width line
                cv2.line(frame,
                         pt1=self.point_finder(face_landmarks, rp1, w, h),
                         pt2=self.point_finder(face_landmarks, rp4, w, h),
                         color=(255,255,255) if ear > self.ear_threshold else (0, 0, 255),
                         thickness = 1)
                
                # writing EAR on the screen 
                cv2.putText(frame,
                            f"EAR: {ear}", 
                            (240, 450),
                            cv2.FONT_HERSHEY_PLAIN,
                            0.8,
                            color = (0,255,0) if ear > self.ear_threshold else (0, 0, 255),
                            thickness=1)
                
                # mouth measures
                mor = self.mouth_open_ratio(face_landmarks, w, h, lips)
                
                # mouth height mid point
                h_mid = self.mid_point_finder(l2, l4, w, h, face_landmarks)
                
                # mouth width mid point
                w_mid = self.mid_point_finder(l1, l3, w, h, face_landmarks)
                
                # mouth width line
                cv2.line(frame, 
                         pt1=self.point_finder(face_landmarks, l1, w, h),
                         pt2=self.point_finder(face_landmarks, l3, w, h),
                         color=(255,255,255) if mor < self.mor_threshold else (0, 0, 255),
                         thickness = 1)
                
                # mouth height line 
                cv2.line(frame, pt1=self.point_finder(face_landmarks, l2, w, h),
                         pt2=self.point_finder(face_landmarks, l4, w, h),
                         color=(255,255,255) if mor < self.mor_threshold else (0, 0, 255),
                         thickness = 1)
                
                # writing the mor on the screen
                cv2.putText(frame,
                            f"MOR: {mor}",
                            (240, 470), 
                            cv2.FONT_HERSHEY_PLAIN, 
                            0.8, 
                            color = (0,255,0) if mor < self.mor_threshold else (0, 0, 255),
                            thickness=1)
                
                
                # drawing all the features
                for point in feature_points:
                    cv2.circle(frame,
                               center = self.point_finder(face_landmarks, point, w, h),
                               radius= 1,
                               color = (0, 255, 0),
                               thickness=-1)
                
                # drawing the approximate mid point
                cv2.circle(frame, center = self.mid_point_finder(h_mid, w_mid, w, h), radius= 1, color = (0, 255, 0), thickness=-1)
                    
        return frame

if __name__ == "__main__":
    
    # reading the config file to get camera path
    config_data = json.load(open("config.json", "r"))
    
    # getting the path from the file
    VIDEO_PATH = config_data['IP_CAM']["tab"]
    
    # getting facial key points, tailored to our needs
    key_points = json.load(open("src/face_mesh_eye_mouth_config.json"))
    
    # initializing the facemesh face detector
    detector = FacexHandDetector()
    
    # capturing frames from camera
    cap = cv2.VideoCapture(VIDEO_PATH)

    # if the video doesn't open
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
        
    #feeding the frames in a loop
    pTime = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame= detector.process_frame(frame,key_points)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.startWindowThread()
        cv2.rectangle(result_frame, (15, 60), (70, 72), color=(0, 0, 0), thickness=-1)
        cv2.putText(result_frame, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    0.8, (255, 255, 255), 1)
        cv2.imshow("Output", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # release the captured frame
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Needed to close the imshow window
