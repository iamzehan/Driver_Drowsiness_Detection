import cv2 

"""
Draw class is single handedly responsible for all kinds of drawing in the image frames.
It uses OpenCV drawing styles to suite our use-case. Besides all the functions are named as self explanatory.
"""


class Draw:
    def __init__(self) -> None:
        pass
    
    def get_color(self, status):
        return (255,255,255) if status else (0, 0, 255)
        
    def draw_eye_features(self, EyeOpen, ear, frame, left_eye_points, right_eye_points):

        lp1, lp4 = left_eye_points[0:4:3]
        rp1, rp4 = right_eye_points[0:4:3]
        
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
                    color=self.get_color(EyeOpen), 
                    thickness = 1)
        
        # right eye width line
        cv2.line(frame,
                    pt1=rp1,
                    pt2=rp4,
                    color=self.get_color(EyeOpen),
                    thickness = 1)
        
        # writing EAR on the screen 
        cv2.putText(frame,
                    f"EAR: {ear}", 
                    (240, 430),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.8,
                    color = self.get_color(EyeOpen),
                    thickness=1)

    def draw_mouth_features(self, Yawning, mor, frame, lips):
        l1, l2, l3, l4 = lips
        # mouth width line
        cv2.line(frame, 
                    pt1=l1,
                    pt2=l3,
                    color=self.get_color(Yawning),
                    thickness = 1)
        
        # mouth height line 
        cv2.line(frame, 
                    pt1=l2,
                    pt2=l4,
                    color=self.get_color(Yawning),
                    thickness = 1)
        
        # writing the mor on the screen
        cv2.putText(frame,
                    f"MOR: {mor}",
                    (240, 450), 
                    cv2.FONT_HERSHEY_PLAIN, 
                    0.8, 
                    color = self.get_color(Yawning),
                    thickness=1)
        
    def draw_intersect(self, finger_tip, index_finger_mcp, frame, nose_to_chin):
        
        nch1, nch2 = nose_to_chin
        
        # drawing the intersecting lines
        cv2.line(frame,
            pt1=nch1,
            pt2=nch2,
            color=(0,0,255),
            thickness = 1)
        
        cv2.line(frame, 
                pt1=index_finger_mcp,
                pt2=finger_tip,
                color=(0,0,255),
                thickness = 1)
        
        # index knuckle
        cv2.circle(frame,
        center= index_finger_mcp,
        radius=2,
        color=(0,255,0),
        thickness=-1)
        
        # any finger tip that intersects
        cv2.circle(frame,
        center= finger_tip,
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
        
    
    def draw_all(self, frame, feature_points, center):
        # drawing all the features
        for point in feature_points:
            cv2.circle(frame,
                        center = point,
                        radius= 1,
                        color = (0, 255, 0),
                        thickness=-1)
            
        # drawing the approximate mid point
        cv2.circle(frame,
                center = center, 
                radius= 1, 
                color = (0, 255, 0),
                thickness=1)
    
    def draw_fps_count(self, frame, fps):
        
        # for visibility of the FPS count
            cv2.rectangle(frame,
                          (15, 60),
                          (70, 72),
                          color=(0, 0, 0),
                          thickness=-1)
            
            # Writing the fps count 
            cv2.putText(frame,
                        f'FPS:{int(fps)}',
                        (20, 70),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.8,
                        (255, 255, 255),
                        1)

    def draw_blink_count(self, frame, blink_count):
        
        # better visibility
        cv2.rectangle(frame,
                          (15, 80),
                          (120, 92),
                          color=(0, 0, 0),
                          thickness=-1)
        
        # Writing the blink count 
        cv2.putText(frame,
                    f'Blink Count:{blink_count}',
                    (20, 90),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.8,
                    (255, 255, 255),
                    1)
    
    def draw_stress_driving(self, frame, eye_blinks):
        
        w = frame.shape[1]
        
        # Writing the blink count 
        if eye_blinks < 21:
            cv2.putText(frame,
                        f'You seem Stressed',
                        (w-150, 40),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.8,
                        (255, 0, 0),
                        1)
        elif eye_blinks >= 21:
            cv2.putText(frame,
                        f'You are Stressed!',
                        (w-150, 40),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.8,
                        (0, 0, 255),
                        1)
        
    def draw_sleepy(self, frame, head_point):
        # for better visibility
        cv2.rectangle(frame,
                          (head_point[0] -8, head_point[1]-10),
                          (head_point[0]+60, head_point[1]+8),
                          color=(0, 0, 255),
                          thickness=-1)
        
        # Writing the Wake up message
        cv2.putText(
                    frame,
                    f"Wake up", 
                    head_point,
                    cv2.FONT_HERSHEY_PLAIN,
                    0.8,
                    color = (255, 255, 255),
                    thickness=1)
    
        