import dlib
import argparse
from skimage import io
from imutils import paths
from scipy.io import loadmat # loading .mat Matlab files
import time

ap = argparse.ArgumentParser()
ap.add_argument("-C", "--Class", required=True, help="Path to Caltech101 class images")
ap.add_argument("-a", "--annotations", required=True, help="Path to Caltech101 annotations")
ap.add_argument("-o", "--output", required=True, help="Path to output detector")
args = ap.parse_args()

print('[INFO]: loading images and annotations...')
options = dlib.simple_object_detector_training_options()
images = []
boxes = []

for imagePath in paths.list_images(args.Class):
	imageID = imagePath[imagePath.rfind("/") + 1:].split("_")[1]
	imageID = imageID.replace(".jpg", "")
	p = "{}/annotation_{}.mat".format(args.annotations, imageID)
	annotations = loadmat(p)["box_coord"]

	# add each annotation to the list of bounding boxes
	bb = [dlib.rectangle(left=long(x), top=long(y), right=long(w), bottom=long(h))
			for (y, h, x, w) in annotations]
	boxes.append(bb)

	# add the imge to the list of images
	images.append(io.imread(imagePath))

# train the object detector
print('[INFO]: training detector...')
tic = time.time()
detector = dlib.train_simple_object_detector(images, boxes, options)
toc = time.time()
print('[INFO]: training took {:.2f} sec...'.format(toc - tic))

# dump the classifer to file
print('[INFO: dump classifer to file')
detector.save(args.output)

# visualize result
win = dlib.image_window()
win.set_image(detector)
dlib.hit_enter_to_continue()

