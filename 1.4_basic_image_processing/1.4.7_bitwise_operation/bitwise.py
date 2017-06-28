import cv2
import numpy as np

# Draw a white rectangle on a black canvas (grayscale image)
rectangle = np.zeros((300, 300), dtype='uint8')
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow("rectangle", rectangle)
cv2.waitKey(0)

circle =  np.zeros((300, 300), dtype='uint8')
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle)
cv2.waitKey(0)

# AND
bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)
cv2.waitKey(0)
# OR
bitwiseOR = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOR)
cv2.waitKey(0)
# XOR
bitwiseXOR = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXOR)
cv2.waitKey(0)
# NOT
bitwiseNOT = cv2.bitwise_not(rectangle)
cv2.imshow("NOT", bitwiseNOT)
cv2.waitKey(0)

# optional: masked xor
mask = np.zeros((300, 300), dtype='uint8')
cv2.rectangle(mask, (0, 0), (200, 200), 255, -1)
masked = cv2.bitwise_xor(rectangle, circle, mask=mask)
cv2.imshow("Masked", masked)
cv2.waitKey(0)