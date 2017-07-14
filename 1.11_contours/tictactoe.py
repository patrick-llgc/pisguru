import cv2

image = cv2.imread("../_images/tictactoe.png")
print(image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(_, contours, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for (i, c) in enumerate(contours):
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    solidity = area / float(hullArea)
    # print(area, hullArea, solidity)

    char = "?"
    if solidity > 0.9:
        char = "0"
    elif solidity > 0.5:
        char = "X"

    if char != "?":
        cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
        cv2.putText(image, char, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
            (0, 255, 0), 4)
    print("contour #{} -- {} -- solidity {:.2f}".format(i, char, solidity))

cv2.imshow("Output", image)
cv2.waitKey(0)