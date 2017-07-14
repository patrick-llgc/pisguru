import cv2
import numpy as np
import argparse

ap = argparse.ArgummentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = ap.parse_args()

image = cv2.imread(args.image)

