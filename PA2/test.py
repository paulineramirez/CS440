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


    contours = cv2.findContours(skin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours[1]:
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue
        #print cv2.contourArea(c)
        x,y,w,h = rect
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)
        cv2.putText(skin,'Skin Detected',(x+w+10,y+h),0,0.3,(0,255,0))
    cv2.imshow("Show",skin)

    # Draw a blue line with thickness of 5 px
    cv2.rectangle(skin,(15,20),(70,50),(0,255,0),5)
    # print("here: " + str(contours[0]))
    # print("there: " + str(contours[1]))
    # print("joe pls: " + str(contours[2]))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('frame2', skin)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()













