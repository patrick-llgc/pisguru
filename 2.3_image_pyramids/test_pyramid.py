from object_detection.helpers import pyramid
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input images')
ap.add_argument('-s', '--scale', type=float, default=1.5, help='Scale factor')
args = ap.parse_args()

image = cv2.imread(args.image)

# display the image pyramid
for (i, layer) in enumerate(pyramid(image, scale=args.scale)):
	cv2.imshow("Layer {}".format(i + 1), layer)
	cv2.waitKey(0)