'''
followed mostly this tutorial 
http://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
http://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

'''

import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

cap = cv2.VideoCapture(0)


def isThumbsUp(w,h):
    #if x in range(175, 250) and y in range(200, 315) and w in range(125, 200) and h in range(225, 320):
    if w in range(100, 200) and h in range(220, 320):
        return True
    return False

def isHeart(w,h):
    if w in range(300, 370) and h in range(85, 140):
        return True
    return False


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # break if there's an issue
    if not ret:
    	break

    # # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = imutils.resize(frame, width = 400)


    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    #print(converted.shape)
    #print(lower.shape)
    #print(upper.shape)
    skinMask = cv2.inRange(converted, lower, upper)
    # # print(skinMask.shape)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    # skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

    #create a matrix of countours (triples of their boundaries)
    contours = cv2.findContours(skin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours[1]: #countours[1] contains the dimensions of detected object
        rect = cv2.boundingRect(c) #find the bounds of detected object (as a rectangle)
        if rect[2] < 100 or rect[3] < 100: continue #JACKIE: might need to play with these
        #print cv2.contourArea(c)
        x,y,w,h = rect
        #print(x,y,w,h) 
        if isThumbsUp(w,h):
            cv2.putText(frame,'THUMBS UP!',(x+w+10,y+h),0,3,(255,255,255)) 

        if isHeart(w,h):
            cv2.putText(frame,'<3!',(x+w+10,y+h),0,3,(255,255,255)) 
        
        #creates rect & text around current frame
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)
        # cv2.putText(frame,'Skin Detected',(x+w+10,y+h),0,3,(255,255,255))

    # Draw a blue line with thickness of 5 px
    cv2.rectangle(skin,(15,20),(70,50),(0,255,0),5)
    # print("here: " + str(contours[0]))
    # print("there: " + str(contours[1]))
    # print("joe pls: " + str(contours[2]))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    # cv2.imshow('frame2', skin)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
















