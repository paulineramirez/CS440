'''
followed mostly this tutorial 
http://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
http://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

'''

from collections import deque
import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

# lower = (0, 48, 80)
# upper = (20, 255, 255)
pts = deque(maxlen=16)

cap = cv2.VideoCapture(0)


def isThumbsUp(w,h):
    #if x in range(175, 250) and y in range(200, 315) and w in range(125, 200) and h in range(225, 320):
    if w in range(100, 200) and h in range(220, 320):
        return True
    return False

def isHeart(w,h):
    if w in range(300, 400) and h in range(55, 170):
        return True
    return False



def isWave(pts):
    if list(pts)[0] is None: 
        return False

    start = pts[0]
    end = pts[len(pts)-1]

    if start is None or end is None:
        return False

    # check if ys in reasonable range with each other
    if np.abs(start[1] - end[1]) in range(50):
        # check distance formula to see if distance is greater than 400 pixels
        if np.abs(start[0] - end[0]) > 250:
            return True

    return False


while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # break if there's an issue
    if not ret:
    	break


    # frame = imutils.resize(frame, width=600)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Start of static gesture
    skinMask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    # cv2.imshow('frame2', skin)

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
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)
    # Start of dynamic gesture

    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(converted, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        # if radius > 10:
        #     # draw the circle and centroid on the frame,
        #     # then update the list of tracked points
        #     cv2.circle(frame, (int(x), int(y)), int(radius),
        #     (0, 255, 255), 2)
        #     cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # cv2.putText(frame,'Motion Detected',(14,20),0,1,(255,255,255)) 

    # show the frame to our screen
    pts.appendleft(center)
    # loop over the set of tracked points
    for i in xrange(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        #print('CURRENT POINT: ' + str(pts[i]))
 
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        # thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
        # cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    # print(pts)
    if isWave(pts):
        cv2.putText(frame,'Hello',(100,100),0,5,(255,255,255)) 
    # Display the resulting frame
    cv2.imshow('frame', frame)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
















