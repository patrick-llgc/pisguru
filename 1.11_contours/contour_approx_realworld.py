import cv2

image = cv2.imread("../_images/receipt_original.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 60, 200)

cv2.imshow("Image", image)
cv2.imshow("Edged", edged)

_, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

for c in cnts:
    peri = cv2.arcLength(c, True) # enclosed contour

    # specify epsilon to be 1% perimeter
    approx = cv2.approxPolyDP(c, 0.01*peri, True)

    print('original {}, approx {}'.format(len(c), len(approx)))

    # cv2.drawContours(image, [approx], -1, (255, 255, 0), 2)
    if len(approx) == 4:
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        

cv2.imshow("Output", image)
cv2.waitKey(0)