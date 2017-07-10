import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Orig", image)

# find and draw contours
(im2, countours, hierarchy) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
clone = image.copy()
cv2.drawContours(clone, countours, -1, (0, 255, 0), 2)
print("Found {} contours".format(len(countours)))

cv2.imshow("All Contours", clone)
cv2.waitKey(0)

