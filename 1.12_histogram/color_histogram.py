import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np

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

# 2D histograms
fig = plt.figure()

ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram (G and B)")
plt.colorbar(p)

ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist)
ax.set_title("2D Color Histogram (G and R)")
plt.colorbar(p)

ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram (G and B)")
plt.colorbar(p)
print('2D histogram shape {} with {} values'.format(
    hist.shape, hist.flatten().shape))

# 3D histogram
fig = plt.figure()
ax = fig.add_subplot(111)
hist = cv2.calcHist([chans[1], chans[0], chans[2]], [0, 1, 2], None, [8, 8, 8],
	[0, 256, 0, 256, 0, 256])
p = ax.imshow(np.hstack(hist), interpolation="nearest")
ax.set_title("3D Color Histogram")
plt.colorbar(p)
print('3D histogram shape {} with {} values'.format(
    hist.shape, hist.flatten().shape))

# display
plt.show()