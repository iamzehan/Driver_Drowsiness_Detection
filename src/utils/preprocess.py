import cv2
import numpy as np

"""
The Enhance class is single handedly responsible for the preprocessing of input frames
From converting color formats, resizing frames to the illumination process, everything is handled here.
"""
class Preprocess:
    def bgr_to_rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def rgb_to_ycbcr(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    
    def resize_image(self, img, frame_size):
        return cv2.resize(img, frame_size)

class Enhance(Preprocess):

    def illumination_enhancement(self, img):
        ycbcr_img = self.rgb_to_ycbcr(img)
        luminance = ycbcr_img[:,:,0]
        n = luminance[0, 0]
        i = luminance[-1, -1]
        
        M = np.sum(luminance) / (n - i)
        
        threshold = 60
        
        if M < threshold:
            enhanced_img = self.histogram_equalization(img)
            return enhanced_img
        else:
            return img
        
    def histogram_equalization(self,img):
        ycbcr_img = self.rgb_to_ycbcr(img)
        y_channel = ycbcr_img[:,:,0]
        
        # Apply Histogram Equalization to the luminance channel
        equ_y_channel = cv2.equalizeHist(y_channel)
        
        # Replace the original luminance channel with the equalized one
        ycbcr_img[:,:,0] = equ_y_channel
        
        # Convert back to RGB
        enhanced_img = cv2.cvtColor(ycbcr_img, cv2.COLOR_YCrCb2RGB)
        
        return enhanced_img