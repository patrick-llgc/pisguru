import cv2
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
args = ap.parse_args()

image = cv2.imread(args.image)
cv2.imshow("Original", image)
cv2.waitKey(0)

# calc hist
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# plot hist
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 256])
plt.plot(hist)

# # Alternative way to do this
# plt.figure()
# hist2 = plt.hist(image[:, :, 0].flatten(), range=[0, 255], bins=255)
# plt.plot(hist2[1][:-1], hist2[0], 'ro')

# normalize
hist /= hist.sum()
plt.figure()
plt.title("Grayscale Histogram (Normalized)")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 256])
plt.plot(hist)

plt.show()

