import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow('Orig', image)
cv2.waitKey(0)

# create rectangular mask for visitors in waterfall.jpg
(h, w) = image.shape[:2]
# mask is generally single channel (grayscale)
mask = np.zeros((h, w), dtype='uint8') 
cv2.rectangle(mask, (344, 209), (428, 287), 255, -1)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
# apply mask. Note the first two args are the same
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Masked Visitors", masked)
cv2.waitKey(0)

# round mask
mask = np.zeros((h, w), dtype='uint8')
cv2.circle(mask, ((344 + 428) // 2, (209 + 287) // 2), 62, 255, -1)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
# apply mask. Note the first two args are the same
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Masked Visitors", masked)
cv2.waitKey(0)