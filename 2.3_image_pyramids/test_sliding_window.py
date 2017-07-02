from object_detection.helpers import pyramid
from object_detection.helpers import sliding_window
import argparse
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input images')
ap.add_argument('-W', '--width', type=int, help='width of sliding window')
ap.add_argument('-H', '--height', type=int, help='height of sliding window')
ap.add_argument('-s', '--scale', type=float, default=1.5, help='Scale factor')
args = ap.parse_args()

image = cv2.imread(args.image)
(winW, winH) = (args.width, args.height)

# window counter
win_number = 0

# loop over the image pyramid 
for layer in pyramid(image, scale=args.scale):
	# loop over the sliding windows
	for (x, y, window) in sliding_window(layer, stepSize=10, windowSize=(winW, winH)):
		# ignore truncated window at boundary
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		clone = layer.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		win_number += 1

		cv2.waitKey(1)
		time.sleep(0.025)

print('Total number of sliding windows {}'.format(win_number))