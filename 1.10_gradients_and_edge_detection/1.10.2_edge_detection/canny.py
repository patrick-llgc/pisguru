import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
ap.add_argument('-l', '--lower_angle', type=float, default=80.0, help='Lower orientation angle')
ap.add_argument('-u', '--upper_angle', type=float, default=100.0, help='Upper orientation angle')
args = ap.parse_args()

image = cv2.imread(args.image)

cv2.imshow("Orig", image)
cv2.waitKey(0)

# Gaussian Blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# compute and compare a wide mie-range and tight threshold for the edges
wide = cv2.Canny(blurred, 10, 200)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 240, 250)

# display
cv2.imshow("Wide Edge Map", wide)
cv2.imshow("Mid Edge Map", mid)
cv2.imshow("Tight Edge Map", tight)
cv2.waitKey(0)

# auto calculate
v = np.median(blurred)
lower = int(max(0, (1-0.33) * v))
upper = int(min(255, (1+0.33) * v))
auto = cv2.Canny(blurred, lower, upper)
cv2.imshow("Auto Edge Map", auto)
cv2.waitKey(0)

