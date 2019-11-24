# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to the image")
args = ap.parse_args()

# load the image and convert it to grayscale
image = cv2.imread(args.image)
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# openCV 3.0+ 
detector = cv2.FastFeatureDetector_create()
kps = detector.detect(gray, None)
 
print("# of keypoints: {}".format(len(kps)))
 
# loop over the keypoints and draw them
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)
 
# show the image
# cv2.imshow("Images", np.hstack([orig, image]))
cv2.imshow("Images", np.hstack([orig, np.dstack([gray] * 3), image]))
cv2.waitKey(0)