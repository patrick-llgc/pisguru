import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to the image")
args = ap.parse_args()

image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)

# erosion
for i in range(3):
	eroded = cv2.erode(gray.copy(), None, iterations=i+1)
	cv2.imshow("Eroded {} times".format(i + 1,), eroded)
	cv2.waitKey(0)

cv2.destroyAllWindows()
# dilation. By default the structuring elemnet is a ones((3, 3))
for i in range(3):
	dilated = cv2.dilate(gray.copy(), None, iterations=i+1)
	cv2.imshow("Dilated {} times".format(i + 1,), dilated)
	cv2.waitKey(0)

# print example structuring element
cv2.destroyAllWindows()
struct_elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
for i in range(3):
	dilated = cv2.dilate(gray.copy(), struct_elem, iterations=i+1)
	cv2.imshow("Dilated {} times".format(i + 1,), dilated)
	cv2.waitKey(0)

# open
cv2.destroyAllWindows()
cv2.imshow("Original", image)
kernelSizes = [(2, 2), (5, 5), (7, 7)]
for kernelSize in kernelSizes:
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	cv2.imshow("Opening: ({}, {})".format(kernelSize[0], kernelSize[1]), opening)
	cv2.waitKey(0)

# close
cv2.destroyAllWindows()
cv2.imshow("Original", image)
for kernelSize in kernelSizes:
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	opening = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
	cv2.imshow("Closing: ({}, {})".format(kernelSize[0], kernelSize[1]), opening)
	cv2.waitKey(0)

# gradient
cv2.destroyAllWindows()
cv2.imshow("Original", image)
for kernelSize in kernelSizes:
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	opening = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
	cv2.imshow("Gradient: ({}, {})".format(kernelSize[0], kernelSize[1]), opening)
	cv2.waitKey(0)

