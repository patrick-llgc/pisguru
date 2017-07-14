import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to image")
args = ap.parse_args()

image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(im2, contours, hierarchy) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clone = image.copy()

green = (0, 255, 0)
white = (255, 255, 255)

# mark centroid of images
for i, c in enumerate(contours):
    # area and perimeter
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True) # True: closed contour
    print("Shape #{} -- area {}, perimeter {}".format(i+1, area, perimeter))

    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print("Shape #{} -- cX {}, cY {}".format(i+1, cX, cY))
    cv2.drawContours(clone, [c], -1, green, 2)
    cv2.circle(clone, (cX, cY), 10, green, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(clone, "#{}".format(i+1), (cX, cY), font, 2, white, 3)

cv2.imshow("Anotated", clone)
cv2.waitKey(0)
clone = image.copy()

# finding bounding box (upright)
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(clone, (x, y), (x+w, y+h), green, 2)

cv2.imshow("Bounding Boxes", clone)
cv2.waitKey(0)
clone = image.copy()

# find bounding box (min area, could be tilted)
for c in contours:
    box = cv2.minAreaRect(c)
    # cv2.circle(clone, (np.int0(box[0][0]), np.int0(box[0][1])), 10, (0,255,0), -1)
    # cv2.circle(clone, (np.int0(box[1][0]), np.int0(box[1][1])), 10, (0,255,255), -1)
    box = np.int0(cv2.boxPoints(box)) # np.int0 is the same as np.int64
    cv2.drawContours(clone, [box], -1, green, 2)

cv2.imshow("Rotated Bounding Boxes", clone)
cv2.waitKey(0)
clone = image.copy()

# find enclosing contour
for c in contours:
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(clone, (int(x), int(y)), int(radius + 2), green, 2) # leave some margin
cv2.imshow("min-enclosing circles", clone)
cv2.waitKey(0)
clone = image.copy()

# fit ellipse
for c in contours:
    if len(c) > 4:
        # need at least 5 points to fit the ellipse
        ellipse = cv2.fitEllipse(c)
        cv2.ellipse(clone, ellipse, green, 2) # leave some margin
cv2.imshow("Ellipses", clone)
cv2.waitKey(0)
clone = image.copy()

