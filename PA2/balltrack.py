import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import deque

# lower =  (123, 11, 41)
# upper = (250, 168, 191)

lower = (0, 48, 80)
upper = (20, 255, 255)


cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# break if there's an issue
	if not ret:
		break

	# resize 
	frame = imutils.resize(frame, width=600)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, lower, upper)
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
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
			(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# # update the points queue
	# pts.appendleft(center)

	# show the frame to our screen
	cv2.imshow("Frame", frame)


	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()