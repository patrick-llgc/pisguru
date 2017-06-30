import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to the image")
args = ap.parse_args()

image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
# blackhat extract features that would be dilated and smaller than the kernel
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
# tophat extract features that would be eroded and smaller than the kernel
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

cv2.imshow("Original", gray)
cv2.imshow("Black Hat", blackhat)
cv2.imshow("Top Hat", tophat)
cv2.waitKey(0)

