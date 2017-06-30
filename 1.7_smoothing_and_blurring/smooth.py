import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)
kernelSizes = [(3, 3), (9, 9), (15, 15)]

# average
for (kX, kY) in kernelSizes:
    blurred = cv2.blur(image, (kX, kY))
    cv2.imshow("Average ({}, {})".format(kX, kY), blurred)
    cv2.waitKey(0)

# gaussian: d is to truncate the infinite sigmaSpace
for (kX, kY) in kernelSizes:
    blurred = cv2.GaussianBlur(image, (kX, kY), 0) # auto calc sigma
    cv2.imshow("Gaussian ({}, {})".format(kX, kY), blurred)
    cv2.waitKey(0)

# median (MUST be square)
for k in (3, 9, 15):
    blurred = cv2.medianBlur(image, k)
    cv2.imshow("Median {}".format(k), blurred)
    cv2.waitKey(0)

# bilateral: reduce noise and keep edges
params = [(11, 21, 7), (11, 1, 7), (11, 61, 39)]
for (diameter, sigmaColor, sigmaSpace) in params:
    blurred = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
    title = "Blurred d={}, sc={}, ss={}".format(diameter, sigmaColor, sigmaSpace)
    cv2.imshow(title, blurred)
    cv2.waitKey(0)


