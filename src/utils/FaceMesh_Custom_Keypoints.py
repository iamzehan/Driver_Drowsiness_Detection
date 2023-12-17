from .Calculations import Calculate
class KeyPoints:
    def __init__(self) -> None:
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LIPS = [78, 13, 308, 14]
        self.NOSE_TO_CHIN = [2, 152]
        self.HANDS = [5, 4, 0, 8, 16, 20]
        self.Calculate = Calculate()
    
    def get_points(self, results, parts, w, h):
        points = [
                self.Calculate.point_finder(landmarks, kp, w, h) 
                    for landmarks in results 
                        for kp in parts
                ]
        return points
    