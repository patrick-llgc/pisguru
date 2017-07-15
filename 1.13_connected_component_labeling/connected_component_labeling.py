import cv2
from skimage.filters import threshold_adaptive
from skimage import measure
import numpy as np

plate = cv2.imread('../_images/license_plate.png')


V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]

# adaptive threshold in skimage
thresh = threshold_adaptive(V, 29, offset=15, method='gaussian').astype(np.uint8) * 255
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Thresh", thresh)

# alternatively in openCV
thresh = cv2.adaptiveThreshold(V, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 29, 15)
cv2.imshow("Thresh cv2", thresh)

# show the images
cv2.imshow("Original", plate)
cv2.imshow("License Plate", V)
cv2.imshow("Thresh", thresh)

# # using grayscale will give somewhat different result
# V = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
# thresh = threshold_adaptive(V, 29, offset=15).astype(np.uint8) * 255
# thresh = cv2.bitwise_not(thresh)
# cv2.imshow("License Plate2", V)
# cv2.imshow("Thresh2", thresh)

# use scikit-image's connected component analysis
labels = measure.label(thresh, neighbors=8, background=0)
n_labels = len(np.unique(labels))
print("INFO: n_labels {} using scikit-image".format(n_labels))

# use openCV's connected component analysis
connectivity = 8
(n_labels, labels, stats, cenntroids) = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
# or without stats
(n_labels, labels) = cv2.connectedComponents(thresh, connectivity, cv2.CV_32S)
print("INFO: n_labels {} using openCV".format(n_labels))

mask = np.zeros(thresh.shape, dtype=np.uint8)
print("INFO: found {} blobs".format(n_labels))

# loop over all components and determine if it is a number
for (i, label) in enumerate(range(n_labels)):
    if label == 0:
        print("INFO: label 0 (background)")
        continue

    # display only the cuurent label
    print("INFO: label {} (foreground)".format(i))
    labelMask = np.zeros(thresh.shape, dtype=np.uint8)
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)

    # add sufficiently large blob to master mask
    if numPixels > 300 and numPixels < 1500:
        mask = cv2.add(mask, labelMask)

    # show label mask
    cv2.imshow("Label", labelMask)
    # cv2.waitKey(0)

cv2.imshow("Large Blobs", mask)


cv2.waitKey(0)