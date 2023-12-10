import cv2
import mediapipe as mp

def hand_detection(img):
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    annotated_image = img.copy()
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return annotated_image, results

if __name__ == "__main__":
    # mediapipe solutions
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5)
    
    #importing image
    path = "Attatchments\A-tired-man-yawning-behind-the-wheel-of-his-car.png"
    img = cv2.imread(path)
    cv2.startWindowThread()
    img, results = hand_detection(img)
    cv2.imshow('img',img)
    for hand_world_landmarks in results.multi_hand_landmarks:
        mp_drawing.plot_landmarks(hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
    cv2.waitKey(0)