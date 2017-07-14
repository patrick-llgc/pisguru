import cv2
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)
cv2.waitKey(0)

# split channel
chans = cv2.split(image)
colors = ('b', 'g', 'r')
# plot hist
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for chan, color in zip(chans, colors):
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	plt.plot(hist, color=color)
	plt.xlim([0, 256])
plt.show()

# 2D histograms