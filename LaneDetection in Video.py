#Video Lane Detection

import cv2
import numpy as np

#Defing a function to draw lines 
def draw_lines( image, lines):
    image= np.copy(image)
    blank_image= np.zeros((image.shape[0], image.shape[1], 3), dtype= np.uint8)
    for li in lines:
        for x1, y1, x2, y2 in li:
            cv2.line(blank_image, (x1,y1), (x2, y2), (255,0,0), 10)
    image= cv2.addWeighted(image, 0.7, blank_image, 0.3, 1)
    return image

#Finding the Region of Interest to remove unwanted region in the video
def ROI( image, vertices):
    #an array which will be used to blackout the unwanted region for better results
    mask= np.zeros_like(image)
    #channel_count= image.shape[2]
    match_mask_color= 255   #*channel_count   #removed this because we're using grayscaled image
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image= cv2.bitwise_and(image, mask)
    return masked_image

def process(img):
     height= img.shape[0]
     width=img.shape[1]
     
     #Will depend upon the video you'll use
     ROIvertices= [(0, height), (width/2, height/5), (width, height)]

     gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
     #Using Canny edge detection mthod
     canny= cv2.Canny(gray, 100 ,200)
     cropped_image= ROI(canny , np.array([ROIvertices], np.int32))
     #using Hough Line Probalistic Transform
     Lines= cv2.HoughLinesP(cropped_image, 6, (np.pi)/180, 10, minLineLength= 100, maxLineGap= 10)
     image_with_lines= draw_lines(img, Lines)
     return image_with_lines


cap= cv2.VideoCapture("Test.mp4")

while (cap.isOpened()):
    ret, frame= cap.read()
    frame= process(frame)
    cv2.imshow("Frame", frame)
    #use Space bar to exit the video loop
    if cv2.waitKey()== 32:
       break

cap.release()
cv2.destroyAllWindows()



















