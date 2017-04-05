from collections import deque
import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt

# lower =  (123, 11, 41)
# upper = (250, 168, 191)

lower = (0, 48, 80)
upper = (20, 255, 255)
pts = deque(maxlen=32)


cap = cv2.VideoCapture(0)

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
		if np.abs(start[0] - end[0]) in range(300):
			return True

	return False


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
		thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
 	# print(pts)
 	if isWave(pts):
 		cv2.putText(frame,'Hello',(14,20),0,1,(255,255,255)) 

	cv2.imshow("Frame", frame)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()