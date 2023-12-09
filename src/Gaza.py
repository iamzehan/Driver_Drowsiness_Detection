import cv2
import json
import numpy as np

# Find the contours of the map of Gaza
def contours(img):
  # convert the image to gray
  imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # resize the image to fit the size of the flag which is 90cm x 150cm by dimension 
  imgray = cv2.resize(imgray, (90,90))
  # get threshold
  ret, thresh = cv2.threshold(imgray, 127, 255, 0)
  # get contours
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # adjust the contour positions to fit a certain location in the flag
  for i, j in enumerate(contours[1]):
    contours[1][i][0][1] = contours[1][i][0][1]+90
  return contours
# Draw the picture
def draw(contours_gaza):
	#Create a blank image
    img = np.zeros((90*3, 150*3, 3),dtype=np.uint8)
	# initialize height and width
    h, w = 0, img.shape[1]
	#iterations
    i = 0
	#RGB color codes for the two flags
    colors = {
	    "Bangladesh": [(0, 106, 78),(244, 42, 65)],
	    "Palestine": [(0,0,0),(255,255,255),(20, 153, 84),(228, 49, 43)]
	    }
	# drawing the flag
    while i<3:
	    # filling up the blank image with colors
        cv2.rectangle(img, (0, h),(w//2, h+90),color=colors["Palestine"][i], thickness = -1)
        cv2.rectangle(img, (w//2, 0),(w, 2*h), color = colors["Bangladesh"][0], thickness = -1)  
        h+=90
        i+=1
	# drawing the triangle in the Palestinian Flag
    cv2.drawContours(img, [np.array([(0,0), (w//3, h//2), (0, h)])], 0, color = colors["Palestine"][-1], thickness=-1)
    cv2.drawContours(img, [contours_gaza[1]], 0, (40, 40, 43), -1)
	# drawing the circle in the Bangladeshi Flag
    cv2.circle(img, center=(w*3//4, h//2), radius = h//3, color = colors["Bangladesh"][1], thickness = -1)
	
	# Converting the colors to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
if __name__ == "__main__":
	# read the image of the map of Gaza Strip
    with open("D:\Documents\Obsidian Vault\config.json", 'r') as config_file:
        config_data = json.load(config_file)
    PATH = config_data["path"]
    gaza = cv2.imread(PATH)
    cv2.startWindowThread()
    contour = contours(gaza)
    bd_x_palestine = draw(contour)
    cv2.imshow('img',bd_x_palestine)
    cv2.waitKey(0)