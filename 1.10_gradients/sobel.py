import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
ap.add_argument('-l', '--lower_angle', type=float, default=80.0, help='Lower orientation angle')
ap.add_argument('-u', '--upper_angle', type=float, default=100.0, help='Upper orientation angle')
args = ap.parse_args()

image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Orig", image)

# compute gradients in 64 bit signed float
gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)

# calculate magnitude and orientation within specified range
mag = np.sqrt(gX ** 2 + gY ** 2)
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

# filter out pixels within lower and upper angle boundaries
grad = np.where(orientation > args.lower_angle, orientation, -1)
grad = np.where(orientation < args.upper_angle, grad, -1)
mask = np.zeros(gray.shape, dtype='uint8')
mask[grad > -1] = 255

cv2.imshow("Gradient within range", mask)
cv2.waitKey(0)

# convert back to unsigned 8-bit and display
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)

# combined the sobel in x and y direction
sobelCombined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

cv2.imshow("SobelX", gX)
cv2.imshow("SobelY", gY)
cv2.imshow("Sobel combined", sobelCombined)
cv2.waitKey(0)