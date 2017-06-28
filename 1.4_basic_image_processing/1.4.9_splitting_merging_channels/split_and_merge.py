import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
args = ap.parse_args()

# show original image
image = cv2.imread(args.image)
cv2.imshow("Orig", image)
cv2.waitKey(0)


# split channels
(blue, green, red) = cv2.split(image)
cv2.imshow("Blue", blue)
cv2.imshow("Green", green)
cv2.imshow("Red", red)
cv2.waitKey(0)
# alternative way to do it in numpy
# red = image[:, :, 2]
# green = image[:, :, 1]
# blue = image[:, :, 0]

# merge channels
merged = cv2.merge((blue, green, red))
cv2.imshow("Merged", merged)
cv2.waitKey(0)
assert np.sum(merged - image) == 0, "Merged is different from original!"
cv2.destroyAllWindows()

# visualize each channel in color
zeros = np.zeros(image.shape[:2], dtype='uint8')
color_blue = cv2.merge((blue, zeros, zeros))
color_green = cv2.merge((zeros, green, zeros))
color_red = cv2.merge((zeros, zeros, red))
# alternative way to do it in numpy
# color_blue2 = np.zeros(image.shape, dtype='uint8')
# color_blue2[:, :, 0] = blue
cv2.imshow("Colored Blue", color_blue)
cv2.imshow("Colored Green", color_green)
cv2.imshow("Colored Red", color_red)
cv2.waitKey(0)
assert np.sum(color_blue + color_green + color_red - image) == 0, "Merged is different from original!"

# quiz
# (blue, green, red) = cv2.split(image)
# print('red: {}'.format(red[94, 180]))
# print('blue: {}'.format(blue[78, 13]))
# print('green: {}'.format(green[5, 80]))