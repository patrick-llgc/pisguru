import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Orig", image)
cv2.imshow("Gray", gray)

# if background is white, invert the image
gray = 255 - gray

# find and draw contours
(im2, contours, hierarchy) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
clone = image.copy()
cv2.drawContours(clone, contours, -1, (0, 255, 0), 2)
print("Found {} contours".format(len(contours)))
cv2.imshow("All Contours", clone)
cv2.waitKey(0)

# loop over contours and draw each of them
clone = image.copy()
cv2.destroyAllWindows()
for (i, c) in enumerate(contours):
    print("Drawing contours #{}".format(i))
    cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Single contour", clone)
    cv2.waitKey(0)

external = 0
if external:
    # retrieve only EXTERNAL contours
    clone = image.copy()
    cv2.destroyAllWindows()
    (im2, contours, hierarchy) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(clone, contours, -1, (0, 255, 0), 2)
    print("Found {} external contours.".format(len(contours)))
    cv2.imshow("All external contours", clone)
    cv2.waitKey(0)

# loop over and show only each contour
clone = image.copy()
cv2.destroyAllWindows()
for c in contours:
    mask = np.zeros(clone.shape, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)

    cv2.imshow("Image", clone)
    cv2.imshow("Mask", mask)
    cv2.imshow("Image + Mask", cv2.bitwise_and(clone, clone, mask=mask))
    cv2.waitKey(0)



