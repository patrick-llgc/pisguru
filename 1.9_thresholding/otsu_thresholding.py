import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Orig", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imshow("Gray", blurred)
cv2.waitKey(0)

(T, threshInv) = cv2.threshold(blurred, 0, 255, 
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Threshold Binary Inverse", threshInv)
print("Threshold found by Otsu: {}".format(T))

output = cv2.bitwise_and(image, image, mask=threshInv)
cv2.imshow("Output", output)
cv2.waitKey(0)