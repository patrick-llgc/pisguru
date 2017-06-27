import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)
cv2.waitKey(0)


# cv2
print("max of 255: {}".format(cv2.add(np.uint8([200]), np.uint8([100]))))
print("min of 0: {}".format(cv2.subtract(np.uint8([50]), np.uint8([100]))))
# np
print("wrap around: {}".format(np.add(np.uint8([200]), np.uint8([100]))))
print("wrap around: {}".format(np.subtract(np.uint8([50]), np.uint8([100]))))

M = np.ones(image.shape, dtype="uint8") * 75
added = cv2.add(image, M)
subtracted = cv2.subtract(image, M)

cv2.imshow("Added", added)
cv2.waitKey(0)
cv2.imshow("Subtracted", subtracted)
cv2.waitKey(0)

# quiz
print("RGB {}".format(added[152, 61][::-1]))