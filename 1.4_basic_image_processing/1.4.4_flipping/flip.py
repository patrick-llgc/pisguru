import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)
cv2.waitKey(0)

# flip horizontally
flipped = cv2.flip(image, 1)
cv2.imshow("Flipped 1", flipped)
cv2.waitKey(0)

# flip vertically
flipped = cv2.flip(image, 0)
cv2.imshow("Flipped 0", flipped)
cv2.waitKey(0)

# flip both 
flipped = cv2.flip(image, -1)
cv2.imshow("Flipped horizontally and vertically", flipped)
cv2.waitKey(0)

# optional: another way to do this: rotate 180 degrees
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 180, 1)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated 180 degrees", rotated)
cv2.waitKey(0)
rotated = imutils.rotate(image, 180)
cv2.imshow("Rotated 180 degrees via imutils", rotated)
cv2.waitKey(0)

# # quiz
# flipped = cv2.flip(image, 1)
# print("RGB {}".format(flipped[235, 259][::-1]))
# flip_rot = imutils.rotate(flipped, 45)
# flip_rot_flip = cv2.flip(flip_rot, 0)
# print("RGB {}".format(flip_rot_flip[189, 441][::-1]))