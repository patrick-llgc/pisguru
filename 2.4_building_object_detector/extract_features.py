# USAGE:
# python extract_features.py -c conf/airplanes.json

from sklearn.feature_extraction.image import extract_patches_2d
from pyimagesearch.object_detection import helpers
#from pyimagesearch.descriptors import HOG
from skimage import feature
from pyimagesearch.utils import dataset
from pyimagesearch.utils import conf
from imutils import paths
from scipy import io
import numpy as np
import progressbar
import argparse
import random
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--conf', required=True, help='path to config file')
args = ap.parse_args()

conf = conf.Conf(args.conf)

# hog = HOG
data = []
labels = []

# feature.hog(image, orientations=conf["orientations"], pixels_per_cell=tuple(conf["pixels_per_cell"]),
#             cells_per_block=tuple(conf["cells_per_block"]), transform_sqrt=conf["normalize"])

# select ground-truth images and select a percentage of them for training
trnPaths = list(paths.list_images(conf["image_dataset"]))
trnPaths = random.sample(trnPaths, int(len(trnPaths) * conf["percent_gt_images"]))
print("[INFO] describing training ROIs...")

# set up the progress bar
widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(trnPaths), widgets=widgets).start()

# loop over training paths
for (i, trnPath) in enumerate(trnPaths):
    # load the image, convert it to grayscale, and extract image ID
    image = cv2.imread(trnPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageID = trnPath[trnPath.rfind("_") + 1:].replace('.jpg', '')  

    # load annotation file
    p = "{}/annotation_{}.mat".format(conf["image_annotations"], imageID)
    bb = io.loadmat(p)["box_coord"][0]
    roi = helpers.crop_ct101_bb(image, bb, padding=conf["offset"], dstSize=tuple(conf["window_dim"]))

    # define sthe list ROIs 
    rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)

    # loop over ROIs
    for roi in rois:
        features = feature.hog(roi, orientations=conf["orientations"], pixels_per_cell=tuple(conf["pixels_per_cell"]),
            cells_per_block=tuple(conf["cells_per_block"]), transform_sqrt=conf["normalize"])
        data.append(features)
        labels.append(1)

    # update progressbar
    pbar.update(i)
pbar.finish()

# grab distraction image path and start another progress bar
dstPaths = list(paths.list_images(conf["image_distractions"]))
print("[INFO] describing distraction ROIs...")
pbar = progressbar.ProgressBar(maxval=conf["num_distraction_images"], widgets=widgets).start()

# loop over distraction images
for i in np.arange(0, conf["num_distraction_images"]):
    # random select a distraction image with replacement, convert to grayscale, 
    # then select random patches
    image = cv2.imread(random.choice(dstPaths))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patches = extract_patches_2d(image, tuple(conf["window_dim"]), 
        max_patches=conf["num_distraction_per_image"])

    for patch in patches:
        features = feature.hog(patch, orientations=conf["orientations"], pixels_per_cell=tuple(conf["pixels_per_cell"]),
            cells_per_block=tuple(conf["cells_per_block"]), transform_sqrt=conf["normalize"])
        data.append(features)
        labels.append(-1)

    pbar.update(i)
pbar.finish()

print("[INFO] dumping features and labels to file...")

dataset.dump_dataset(data, labels, conf["features_path"], "features")
print("size of labels {}".format(len(labels)))
print("size of data {} {}".format(len(data), len(data[0])))




