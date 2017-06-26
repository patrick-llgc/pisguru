import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--image", "-i", required=True, help="Path to image file")

# initialize canvas as 300x300 with 3 channels RGB, with a black (0,0,0) background
# uint8 ensures the channel values are [0, 255]
canvas = np.zeros((300, 300, 3), dtype='uint8')

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
white = (255, 255, 255)

# draw a green line from top-left to bottom-right
cv2.line(canvas, (0, 0), (300, 300), green)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# draw a red line from top-right to bottom-left
cv2.line(canvas, (0, 300), (300, 0), red, 3)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# draw a green 50x50 rectangle 
cv2.rectangle(canvas, (10, 10), (60, 60), green)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# draw a red rectangle of 5 pixel thick
cv2.rectangle(canvas, (50, 200), (200, 255), red, 5)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# draw a filled blue rectangle
cv2.rectangle(canvas, (200, 50), (225, 125), blue, -1)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# reset canvas
canvas = np.zeros((300, 300, 3), dtype='uint8')
(h, w) = canvas.shape[:2]
(centerX, centerY) = (w // 2, h // 2)

# draw a bullseye
for r in range(0, 175, 25):
    cv2.circle(canvas, (centerX, centerY), r, white)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# draw random circles with random color
for _ in range(25):
    radius = np.random.randint(5, high=200)
    # np.array.tolist() and list(np.array) is different
    # in that the element in each unit is different
    color = np.random.randint(0, 256, size=(3, )).tolist()
    center = np.random.randint(0, 300, size=(2, ))
    cv2.circle(canvas, tuple(center), radius, color, -1)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

# load an exisiting image and use as canvas
image = cv2.imread("waterfall.JPG")
# Note: points are in the order of (x, y)
cv2.line(image, (2, 260), (280, 100), red, 5)
cv2.circle(image, (400, 260), 50, green, 2)
cv2.rectangle(image, (400, 100), (450, 150), blue, -1)

cv2.imshow("waterfall", image)
cv2.waitKey(0)


