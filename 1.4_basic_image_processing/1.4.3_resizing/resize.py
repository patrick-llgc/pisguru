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

# keep the aspect ratio when resizing
# resize so that the width is 150
new_w = 150
new_h = int(float(new_w) / w * h)
dim = (new_w, new_h)
# cv2.INTER_AREA is preferred for image decimation (downsampling)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized_area", resized)
cv2.waitKey(0)
# lanczos is better when scaling up, but in practise rarely used
# cv2.INTER_CUBIC is more often used
resized = cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)
cv2.imshow("Resized_lanczos", resized)
cv2.waitKey(0)

# resize so that the height is 50
new_h = 50
new_w = int(float(new_h) / h * w)
dim = (new_w, new_h)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized", resized)
cv2.waitKey(0)

# use imutils convenience function
resized = imutils.resize(image, width=150)
cv2.imshow("Resized via imutils", resized)
cv2.waitKey(0)
resized = imutils.resize(image, height=100, inter=cv2.INTER_NEAREST)
cv2.imshow("Resized via imutils", resized)
cv2.waitKey(0)

# quiz
resized = imutils.resize(image, width=100, inter=cv2.INTER_NEAREST)
print("RGB: {}".format(resized[74, 20][::-1]))
resized = imutils.resize(image, width=2 * w, inter=cv2.INTER_CUBIC)
print("RGB: {}".format(resized[367, 170][::-1]))
