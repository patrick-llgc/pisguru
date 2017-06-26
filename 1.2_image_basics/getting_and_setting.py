import argparse
import cv2
import numpy as np

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = ap.parse_args()

# load the image, get dimension and show image
image = cv2.imread(args.image)
(h, w) = image.shape[:2]
cv2.imshow("Original", image)

# image is a 3D numpy array. Access bgr values at coordinate (100, 100)
(b, g, r) = image[100, 100] 
print("Pixel at (0, 0) - Red {}, Green {}, Blue {}".format(r, g, b))

# newimage = image only creates a reference
newimage = image.copy()
# make it blue (NOT red)
newimage[100, 100] = (0, 0, 255)
(b, g, r) = newimage[0, 0] 
print("Pixel at (0, 0) - Red {}, Green {}, Blue {}".format(r, g, b))

cv2.imshow("Modified image", np.hstack((image, newimage)))
cv2.waitKey(0)

(cX, cY) = (w // 2, h // 2)
tl = image[0:cY, 0:cX]
tr = image[:cY, cX:]
bl = image[cY:, :cX]
br = image[cY:, cX:]
cv2.imshow("Top-Left Corner", tl)
cv2.imshow("Top-Right Corner", tr)
cv2.imshow("Btm-Left Corner", bl)
cv2.imshow("Btm-Right Corner", br)
cv2.waitKey(0)

newimage[0:cY, 0:cX] = (0, 255, 0)
cv2.imshow("Block Green", newimage)
cv2.waitKey(0)