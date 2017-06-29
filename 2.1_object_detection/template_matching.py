import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
ap.add_argument('-t', '--template', required=True, help='Template to be matched')
ap.add_argument('-v', '--visualize', help='Visualize the overlaid image')
args = ap.parse_args()

# read in image and template
image = cv2.imread(args.image)
orig = image.copy()
template = cv2.imread(args.template)
(h, w) = template.shape[:2]

R = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
R_norm = np.uint8(R / np.max(R) * 256) - 1
cv2.imshow("Results", R)
cv2.waitKey(0)

(minVal, maxVal, minLoc, (x, y)) = cv2.minMaxLoc(R)
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Detected", image)
cv2.imshow("Template", template)
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.waitKey(0)

print('Best match location: x {}, y {}'.format(x, y))

# optinal: overlay
if args.visualize is not None:
	merged = orig
	R = (R - np.min(R)) / (np.max(R) - np.min(R)) * 128
	(R_h, R_w) = R.shape
	merged[:R_h, :R_w, 0] = np.uint8(R) + merged[:R_h, :R_w, 0] * 0.2
	merged[:R_h, :R_w, 1] = np.uint8(R) + merged[:R_h, :R_w, 1] * 0.2
	merged[:R_h, :R_w, 2] = np.uint8(R) + merged[:R_h, :R_w, 2] * 0.2
	cv2.imshow("Detected + Template", merged)
	cv2.waitKey(0)
	print(np.max(R))

