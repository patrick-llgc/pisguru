import cv2
import numpy as np
import argparse
import matplotlib.pylab as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help='Path to the image')
args = ap.parse_args()

image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)

cv2.imshow("Original", gray)
cv2.imshow("Histogram Equalization", eq)
cv2.waitKey(0)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([eq], [0], None, [256], [0, 256])

# visualize histograms
plt.figure()
plt.plot(hist, 'b')
plt.plot(hist2, 'r')
plt.title('Histograms')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')

# visualize cumulative sum of histograms
plt.figure()
plt.plot(np.cumsum(hist), 'b')
plt.plot(np.cumsum(hist2), 'r')
plt.title('Cumsum of Histograms')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')
plt.show()