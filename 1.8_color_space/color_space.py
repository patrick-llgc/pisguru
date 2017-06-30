import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = ap.parse_args()

# BGR space
image = cv2.imread(args.image)
cv2.imshow("Original", image)
for name, ch in zip(['B', 'G', 'R'], cv2.split(image)):
    cv2.imshow(name, ch)
cv2.waitKey(0)
cv2.destroyAllWindows()

# convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
for name, ch in zip(['H', 'S', 'V'], cv2.split(hsv)):
    cv2.imshow(name, ch)
cv2.waitKey(0)
cv2.destroyAllWindows()

# convert to L*A*B*
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB", lab)
for name, ch in zip(['L*', 'A*', 'B*'], cv2.split(lab)):
    cv2.imshow(name, ch)
cv2.waitKey(0)
cv2.destroyAllWindows()

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
