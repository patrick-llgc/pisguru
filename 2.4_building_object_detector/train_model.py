# USAGE:
# python train_model.py --conf conf/airplanes.json

from pyimagesearch.utils import dataset
from pyimagesearch.utils import conf
from sklearn.svm import SVC
import numpy as np
import argparse
try:
    import cPickle as pickle
except:
    import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help='path to configuration file')
ap.add_argument("-n", "--hard-negatives", type=int, default=-1, 
    help='flag indicating whether or not hard negatives should be used')
args = ap.parse_args()

print("[INFO] loading dataset...")
conf = conf.Conf(args.conf)
data, labels = dataset.load_dataset(conf["features_path"], "features")

# check if hard negative flag is used
if args.hard_negatives > 0:
    print("[INFO] loading hard negatives ...")
    hardData, hardLabels = dataset.load_dataset(conf["features_path"], "hard_negatives")
    data = np.vstack([data, hardData])
    labels = np.hstack([labels, hardLabels])

# train the classifier
print("[INFO] training classifier...")
model = SVC(kernel="linear", C=conf["C"], probability=True, random_state=1)
model.fit(data, labels)

# serialize and dump classifier to file
print("[INFO] dumping classifier...")
f = open(conf["classifier_path"], "wb")
f.write(pickle.dumps(model))
f.close()
