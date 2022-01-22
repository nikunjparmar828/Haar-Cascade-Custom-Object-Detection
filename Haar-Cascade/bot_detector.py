import cv2
import numpy as np

# preprocessing before feeding the frame to Haar-cascade to reduce the 
# Computation 
def preprocessing(frame):
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame= cv2.GaussianBlur(frame,(3,3),0)
    
    red_upper_bound=(5,255,255)
    red_lower_bound=(0,140,140)
    red_mask=cv2.inRange(frame, red_lower_bound,red_upper_bound)
    
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    morphed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow("mask", morphed)
    
    contours, hierarchy = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=list()

    for contour in contours:
        if cv2.contourArea(contour)>200:
            x,y,w,h=cv2.boundingRect(contour)
            boxes.append((x,y,w,h))

    return boxes        

def main():

    '''
    Well, first, we need to collect the image from an appropriate source and then
    pre process the frame. The result of the preprocessed frame should be some n smaller
    rectangular regions, Grayscaled which MAY contain robot.

    Then we need to detect the robot in the pre processed data


    first initialize the detector object
    then do the preprocessing on the frame

    Preprocessing:
    1. Convert frame to hsv.
    2. filter out the R channel.
    3. Apply Openning
    4. Find the large contours. what contours to include 
    will be given by a parameter we need to define
    5. 
    '''
    vid = cv2.VideoCapture(0)
    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("frame", frame)
        boxes=preprocessing(frame)
        for box in boxes:
            x,y,w,h= box
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=='__main__':
    main()
