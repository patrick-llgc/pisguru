import argparse
import numpy as np
import imutils
import cv2

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)
(h, w) = image.shape[:2]
cv2.waitKey(0)

# translate an image using warpAffine
# it is defined by a numpy array
# 	[[alpha, beta, (1-alpha)*centerX - beta*centerY],
#    [-beta, alpha, beta*ceterX + (1-alpha)*centerY]]
# where alpha = scale * cos(angle)
#        beta = scale * sin(angle)
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, 90, 1)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated 90 degrees anti-clockwise wrt center", rotated)
cv2.waitKey(0)

center = (0, 0)
M = cv2.getRotationMatrix2D(center, 30, 1)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated 30 degrees anti-clockwise wrt top left corner", rotated)
cv2.waitKey(0)

# use predefined imutils.rotate() convenience function
rotated = imutils.rotate(image, 180, (50, 50))
cv2.imshow("Rotated by 180 degrees", rotated)
cv2.waitKey(0)

# quiz
rotated = imutils.rotate(image, -30)
print('RGB: {}'.format(rotated[254, 335][::-1]))
rotated = imutils.rotate(image, 110)
print('RGB: {}'.format(rotated[136, 312][::-1]))
M = cv2.getRotationMatrix2D((50, 50), 88, 1)
rotated = cv2.warpAffine(image, M, (w, h))
print('RGB: {}'.format(rotated[10, 10][::-1]))
