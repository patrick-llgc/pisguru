import cv2
import numpy as np

image = cv2.imread("../_images/tetris_blocks.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

_, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hullImage = np.zeros(gray.shape, dtype=np.uint8)
for i, c in enumerate(cnts):
    # area
    area = cv2.contourArea(c)

    # aspect ratio
    x, y, w, h = cv2.boundingRect(c)
    aspectRatio = w / float(h)

    # extent 
    extent = area / float(w * h)

    # solidity
    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    solidity = area / float(hullArea)

    cv2.drawContours(hullImage, [hull], -1, 255, -1)
    cv2.drawContours(image, [c], -1, (0, 0, 0), 3)
    shape = ""

    # shape determination
    if aspectRatio > 0.98 and aspectRatio < 1.02:
        shape = "SQUARE"
    elif aspectRatio > 3:
        shape = "RECTANGLE"
    elif extent < 0.65:
        shape = "L-piece"
    elif solidity > 0.8:
        shape = "Z-piece"

    cv2.putText(image, shape, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, .5, 
        (0, 255, 0), 2)

    print("Contour #{} -- aspect ratio={:.2f}, extent={:.2f}, solidity={:.2f}".format(
        i+1, aspectRatio, extent, solidity))

    cv2.imshow("Convex Hull", hullImage)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

