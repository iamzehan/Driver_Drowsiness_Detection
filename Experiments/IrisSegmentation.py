import time
import cv2 as cv
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


cap = cv.VideoCapture('Videos/3.mp4')
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height)
result = cv.VideoWriter('Videos/Mask/eye_tracker.mp4',  
                         cv.VideoWriter_fourcc(*'MP4V'), 
                         10, size) 
with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            # eye lens
            cv.circle(frame, center_left, int(l_radius), (0, 255, 0), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (0, 255, 0), 1, cv.LINE_AA)
            # eye balls
            cv.circle(frame, center_left, int(l_radius)*3, (0, 255, 0), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius)*3, (0, 255, 0), 1, cv.LINE_AA)
            #center of the eye/pupils
            cv.circle(frame, center_left, 1, (0, 0, 255), 1, cv.LINE_AA) # center of the eye left
            cv.circle(frame, center_right, 1, (0, 0, 255), 1, cv.LINE_AA) # center of the eye right
            
            time.sleep(0.0001)
            
            # Now to show it in a separate mask 
            mask = np.zeros((img_h, img_w, 3), dtype=np.uint8) 
            cv.circle(mask, center_left, int(l_radius)*3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(mask, center_right, int(r_radius)*3, (255, 255, 255), -1, cv.LINE_AA)
            # eye lens
            cv.circle(mask, center_left, int(l_radius), (255, 255, 0), -1, cv.LINE_AA)
            cv.circle(mask, center_right, int(r_radius), (255, 255, 0), -1, cv.LINE_AA)
            cv.circle(mask, center_left, int(l_radius), (0, 0, 0), 1, cv.LINE_AA)
            cv.circle(mask, center_right, int(r_radius), (0, 0, 0), 1, cv.LINE_AA)
            #center of the eye/pupils
            cv.circle(mask, center_left, 3, (0, 0, 0), -1, cv.LINE_AA) # center of the eye left
            cv.circle(mask, center_right, 3, (0, 0, 0), -1, cv.LINE_AA) # center of the eye right
            
        cv.imshow('img', frame)
        cv.imshow('Mask - ©️ Ziaul Karim',mask)
        result.write(mask)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
cap.release()
cv.destroyAllWindows()