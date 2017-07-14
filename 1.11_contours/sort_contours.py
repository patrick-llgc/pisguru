import numpy as np
import cv2
import argparse

def sort_contours(cnts, method="left-to-right"):
	reverse = False
	i = 0

	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True

	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	# sort by bounding boxes
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	print(boundingBoxes)
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	print(boundingBoxes)
	return (cnts, boundingBoxes)

def find_center(contour):
	M = cv2.moments(contour)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	return cX, cY

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-m", "--method", required=True, help="Sorting method")
args = ap.parse_args()

image = cv2.imread(args.image)
accumEdged = np.zeros(image.shape[:2], dtype=np.uint8)

# loop over bgr channels
for chan in cv2.split(image):
	chan = cv2.medianBlur(chan, 11)
	edged = cv2.Canny(chan, 50, 200)
	accumEdged = cv2.bitwise_or(accumEdged, edged)

cv2.imshow("Edge Map", accumEdged)
cv2.waitKey(0)


# find contours and pick the top 5 largest
cnts, _ = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# draw original order
orig = image.copy()
for i, c in enumerate(cnts):
	x, y = find_center(c)
	cv2.putText(orig, "#{}".format(i), (x-20, y), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 255, 0), 2)
cv2.imshow("Unsorted", orig)
cv2.waitKey(0)

# sorted according to coordinate
orig = image.copy()
cnts, boundingBoxes = sort_contours(cnts, args.method)
for i, c in enumerate(cnts):
	x, y = find_center(c)
	cv2.putText(orig, "#{}".format(i), (x-20, y), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 255, 0), 2)
cv2.imshow("Sorted", orig)
cv2.waitKey(0)

