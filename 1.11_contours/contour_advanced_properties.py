import cv2

image = cv2.imread("../_images/more_shapes_example.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, cnts, _ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, c in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(c)
    aspectRatio = w / float(h)

    area = cv2.contourArea(c)
    extent = area / float(w * h)

    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    solidity = area / float(hullArea)

    print("aspectRatio: {:.2f}, solidity: {:.2f}, extent: {:.2f}".format(
        aspectRatio, solidity, extent))
    cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
    cv2.imshow("image", image)
    cv2.waitKey(0)