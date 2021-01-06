#Road Lane Detection

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Making afuntion to draw lines on the image

def draw_lines( image, lines):
    image= np.copy(image)
    blank_image= np.zeros((image.shape[0], image.shape[1], 3), dtype= np.uint8)
    for li in lines:
        for x1, y1, x2, y2 in li:
            cv2.line(blank_image, (x1,y1), (x2, y2), (255,0,0), 10)
    image= cv2.addWeighted(image, 0.7, blank_image, 0.3, 1)
    return image

#Defining a function to detect Region Of Interest

def ROI( image, vertices):
    #creating an array which is all black (to blackout the unwanted region)
    mask= np.zeros_like(image)
    #channel_count= image.shape[2]  #Deleted this bcoz we'll use a grayscale image
    match_mask_color= 255   #*channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image= cv2.bitwise_and(image, mask)
    return masked_image

img= cv2.imread("lane1.jpg")

#Converting because opencv uses BGR method// pyplot uses RGB method
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
height= img.shape[0]
width=img.shape[1]

#detecting the vertices-- will depend upon the image you use
#this will be used to lackout unwanted region in ROI()
ROIvertices= [(0, height), (width/2, height/1.9), (width, height)]

gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#using Canny Edge Detection to detect 
canny= cv2.Canny(gray, 100 ,200)
cropped_image= ROI(canny , np.array([ROIvertices], np.int32))

#Using Hough Line Transform --put parameters of your choice and experiment
Lines= cv2.HoughLinesP(cropped_image, 6, (np.pi)/180, 10, minLineLength= 100, maxLineGap= 10)
#Final Result
image_with_lines= draw_lines(img, Lines)

#Showing Images using Pyplot 
plt.imshow(cropped_image)
plt.show()
plt.imshow(image_with_lines)
plt.show()





















