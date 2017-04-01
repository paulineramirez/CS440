'''
followed mostly this tutorial 
http://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
http://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

'''

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
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    # skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('frame2', skin)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()