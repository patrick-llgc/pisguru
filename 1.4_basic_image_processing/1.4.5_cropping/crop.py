import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)
cv2.waitKey(0)

# use ../waterfall.jpg as example and supply a priori knowledge
# crop out a particular visitor
visitor = image[209:287, 344:428]
cv2.imshow("Visitor", visitor)
cv2.waitKey(0)
# and the whole crowd
crowd = image[209:287, 76:428]
cv2.imshow("Crowd", crowd)
cv2.waitKey(0)

# # quiz
# cropped = image[173:235, 13:81]
# cv2.imshow("Cropped1", cropped)
# cv2.waitKey(0)
# cropped = image[124:212, 225:380]
# cv2.imshow("Cropped2", cropped)
# cv2.waitKey(0)
# cropped = image[85:250, 85:220]
# cv2.imshow("Cropped3", cropped)
# cv2.waitKey(0)
# cropped = image[90:450, 0:290]
# cv2.imshow("Cropped4", cropped)
# cv2.waitKey(0)
