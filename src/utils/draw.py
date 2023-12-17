import cv2 

class Draw:
    def __init__(self) -> None:
        self.mor_threshold = 0.6
        self.ear_threshold = 0.25
    
    def draw_eye_property(self, ear, frame, left_eye_points, right_eye_points):

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
        return frame
    def draw_mouth_property(self, mor, frame, lips):
        l1, l2, l3, l4 = lips
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
        
        
        return frame
    def draw_intersect(self, i, frame, nose_to_chin, hand_points):
        nch1, nch2 = nose_to_chin
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
        
        return frame