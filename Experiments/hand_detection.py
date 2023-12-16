import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self):
        # mediapipe solutions
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5)
    def hand_detection(self, img):
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # we copy the image for annotation
        annotated_image = img.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
        return annotated_image, results
    def draw_landmarks_plot(self, results):
        for hand_world_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.plot_landmarks(hand_world_landmarks, self.mp_hands.HAND_CONNECTIONS, azimuth=5)
if __name__ == "__main__":
    #importing image
    path = "Attatchments\A-tired-man-yawning-behind-the-wheel-of-his-car.png"
    img = cv2.imread(path)
    detect = HandDetector()
    cv2.startWindowThread()
    img, results = detect.hand_detection(img)
    cv2.imshow('img',img)
    detect.draw_landmarks_plot(results)
    cv2.waitKey(0)