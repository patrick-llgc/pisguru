from scipy import io
import numpy as np
import argparse
import glob
# from pyimagesearch.utils import Conf
from pyimagesearch.utils import conf

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=False, help="path to the config file")
args = ap.parse_args()

print('args.conf={}'.format(args.conf))
conf = conf.Conf(args.conf)
widths = []
heights = []

# loop over all annotation paths
for p in glob.glob(conf["image_annotations"] + "/*.mat"):
    (y, h, x, w) = io.loadmat(p)["box_coord"][0]
    widths.append(w - x)
    heights.append(h - y)

avgWidth, avgHeight = np.mean(widths), np.mean(heights)
print('[INFO]: avg. width: {:.2f}'.format(avgWidth))
print('[INFO]: avg. height: {:.2f}'.format(avgHeight))
print('[INFO]: aspect ratio: {:.2f}'.format(avgWidth / avgHeight))
