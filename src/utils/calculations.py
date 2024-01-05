import math

"""
function mid_point_finder, returns the midpoint between two points.
function length_calculation, returns the distance between two points.
function eye_aspect_ratio, calculates and returns the Eye Aspect Ratio of an eye.
function mouth_open_ratio, calculates and return Mouth Open Ratio.
function mid_mouth_open_ratio, calculates and returns the Approximate Mouth Open ratio.
function calculate_slope, calculates the slope of a line
function check_intersection, checks the intersection between to lines and returns a boolean value.

"""

class Calculate:
    
    # mid point finder
    def mid_point_finder(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return (x2 + x1)//2, (y2+y1)//2
    
    # calculate length between two points
    def length_calculation(self, point1, point2):
        x_1, y_1 = point1
        x_2, y_2 = point2
        return math.sqrt((x_1-x_2)**2 + (y_1 - y_2)**2)

    # Function to calculate Eye Aspect Ratio (EAR) based on facial landmarks
    def eye_aspect_ratio(self, args):
        
        # Extracting coordinates of the eyes
        p1, p2, p3, p4, p5, p6 = args
        
        # Calculating Lengths
        A = self.length_calculation(p2, p6)
        B = self.length_calculation(p3, p5)
        C = self.length_calculation(p1, p4)
        
        # calculating EAR
        ear = (A+B) / (2.0*C) 
        return ear

    # Mouth Open Ratio (MOR)
    def mouth_open_ratio(self, args):
        l1,l2,l3,l4 = args 
        mouth_w, mouth_h = self.length_calculation(l1, l3),\
            self.length_calculation(l2, l4)
        return mouth_h/mouth_w
    
    def mid_mouth_open_ratio(self, h_mid, w_mid, args):
        # all mouth points
        l1, l2, l3, l4 = args
        # mouth height mid point
        mid = self.mid_point_finder(h_mid, w_mid)

        approx_h = self.length_calculation(l2, mid) \
            + self.length_calculation(l4, mid)
        approx_w = self.length_calculation(l1, mid) \
            + self.length_calculation(l3, mid)
        return approx_h/approx_w
        
    def calculate_slope(self, x1, y1, x2, y2):
        if (x2-x1)>0:
            return (y2 - y1) / (x2 - x1)
        else:
            return 0

    def check_intersection(self, line1, line2):
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        m1 = self.calculate_slope(x1, y1, x2, y2)
        m2 = self.calculate_slope(x3, y3, x4, y4)

        # Check if lines are parallel
        if m1 == m2:
            return False

        # Calculate the y-intercepts
        b1 = y1 - m1 * x1
        b2 = y3 - m2 * x3

        # Calculate the x-coordinate of the intersection point
        x_intersect = (b2 - b1) / (m1 - m2)

        # Check if the intersection point is within the line segments
        if min(x1, x2) <= x_intersect <= max(x1, x2) and min(x3, x4) <= x_intersect <= max(x3, x4):
            return True
        else:
            return False