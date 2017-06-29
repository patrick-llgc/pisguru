import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Orig", image)
cv2.waitKey(0)

# blurring
kernel = np.ones((3, 3), np.float32) / 9
blurred = cv2.filter2D(image, -1, kernel)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# outline
kernel = np.array([[-1, -1, -1],
				   [-1,  8, -1],
				   [-1, -1, -1]])
outlined = cv2.filter2D(image, -1, kernel)
cv2.imshow("Outline", outlined)
cv2.waitKey(0)


# horizontal sobel
kernel = np.array([[-1, -2, -1],
				   [ 0,  0,  0],
				   [ 1,  2,  1]])
sobel_h = cv2.filter2D(image, -1, kernel)
cv2.imshow("Horizontal Sobel", np.abs(sobel_h))
cv2.waitKey(0)

kernel = np.array([[ 1,  0, -1],
				   [ 2,  0, -2],
				   [ 1,  0, -1]])
sobel_v = cv2.filter2D(image, cv2.CV_16S, -kernel)
cv2.imshow("Vertical Sobel", np.uint8(np.abs(sobel_v)))
cv2.waitKey(0)

# call Sobel() directly
sobel_v2 = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
cv2.imshow("Vertical Sobel2", np.uint8(np.abs(sobel_v2)))
cv2.waitKey(0)
assert np.sum(sobel_v - sobel_v2) == 0

# sharpen
kernel = np.array([[ 0,  -1,  0],
				   [-1,   5, -1],
				   [ 0,  -1,  0]])
sharpen = cv2.filter2D(image, -1, kernel)
cv2.imshow("Sharpen", sharpen)
cv2.waitKey(0)