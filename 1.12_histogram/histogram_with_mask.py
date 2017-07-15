import cv2
import numpy as np
import matplotlib.pylab as plt
import argparse

def plot_histogram(image, title, mask=None):
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 255])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

# read in file
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help='Path to the image')
args = ap.parse_args()

# display original image and plot histogram
image = cv2.imread(args.image)
cv2.imshow("Original", image)
plot_histogram(image, "Histogram of original image")

# display masked image and plot histogram
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, (200, 1), (300, 20), 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Masked", masked)
plot_histogram(image, "Histogram for Masked Image", mask=mask)
plt.show()