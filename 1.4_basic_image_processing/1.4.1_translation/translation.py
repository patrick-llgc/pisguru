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
# 	[[1, 0, shiftX],[0, 1, shiftY]]
M = np.float32([[1, 0, 25], [0, 1, 50]])
shifted = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Shifted down 50 and right 25 pixels", shifted)
cv2.waitKey(0)

M = np.float32([[1, 0, -50], [0, 1, -90]])
shifted = cv2.warpAffine(image, M, (w//2, h//2))
cv2.imshow("Shifted up 90 and left 50 pixels", shifted)
cv2.waitKey(0)


