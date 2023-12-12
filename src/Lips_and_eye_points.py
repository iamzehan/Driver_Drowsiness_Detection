import cv2
import json
import numpy as np
import math
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
        # threshold for mouth opening ratio
        self.mor_threshold = 0.6
        
    def point_finder(self, face_landmarks, point, w, h):
        return (int(face_landmarks.landmark[point].x*w),int(face_landmarks.landmark[point].y*h))
    
    def length_calculation(self, face_landmarks, point1, point2, w, h):
        x_1, y_1 = self.point_finder(face_landmarks, point1, w, h)
        x_2, y_2 = self.point_finder(face_landmarks, point2, w, h)
        return math.sqrt((x_1-x_2)**2 + (y_1 - y_2)**2)
    
    def mouth_open_ratio(self, height, width):
        return round(height/width, 2)
    
    def process_frame(self, frame):
        # Resize frame for faster processing (adjust as needed)
        frame = cv2.resize(frame, (480, 480))

        # Convert to RGB for MediaPipe models
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #get frame height, width and channel
        h, w, c = frame.shape        
        
        # Face Landmarks
        results_face = self.face_mesh.process(rgb_frame)
        
        #coordinates for inner right eye 
        right_eye_points = [33, 160, 158, 133, 153, 144]
        rp1, rp2, rp3,rp4, rp5, rp6 = right_eye_points
        
        # coordinates for inner left eye
        left_eye_points = [363, 385, 387, 263, 373, 380]
        lp1, lp2, lp3, lp4, lp5, lp6 = left_eye_points
        # coordinates point for inner lips
        lips = [78, 13, 308, 14]
        l1, l2, l3, l4 = lips
        feature_points = left_eye_points+right_eye_points+lips
        
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                
                # mouth measures
                mouth_w, mouth_h = self.length_calculation(face_landmarks, l1, l3, w, h), self.length_calculation(face_landmarks, l2, l4, w, h)
                mor = self.mouth_open_ratio(mouth_h, mouth_w)
                color = (255,255,255) if mor < self.mor_threshold else (0, 0, 255)
                
                # mouth width line
                cv2.line(frame, pt1=self.point_finder(face_landmarks, l1, w, h), pt2=self.point_finder(face_landmarks, l3, w, h),color=color, thickness = 1)
                # mouth height line 
                cv2.line(frame, pt1=self.point_finder(face_landmarks, l2, w, h), pt2=self.point_finder(face_landmarks, l4, w, h),color=color, thickness = 1)
                
                # writing the mor on the screen
                cv2.putText(frame,f"MOR: {mor}", (240, 470), cv2.FONT_HERSHEY_PLAIN,
                                    0.8, color = (0,0,0) if mor < self.mor_threshold else (0, 0, 255), thickness=1)
                
                # drawing the landmarks with green dots/circles
                for id, lm in enumerate(face_landmarks.landmark):
                    if id in feature_points:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, center = (x, y), radius= 1, color = (0, 255, 0), thickness=-1)
                    
        return frame

if __name__ == "__main__":
    with open("config.json", "r") as config_file:
        config_data = json.load(config_file)
    VIDEO_PATH = config_data['IP_CAM']["phone"]
    
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    detector = FacexHandDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame= detector.process_frame(frame)
        cv2.startWindowThread()
        cv2.imshow("Output", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Needed to close the imshow window
