import argparse
import cv2
import numpy as np
import matplotlib.pylab as plt
import os
import glob

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image file")
ap.add_argument("-o", "--output", required=False, help="Path to output file")
ap.add_argument("--loop", help="Number of load-save loops")
args = ap.parse_args()

# load image and show basic info
image = cv2.imread(args.image)  
print("file name: {}".format(args.image))
print("width: {}".format(image.shape[0]))
print("length: {}".format(image.shape[1]))

# show the image
cv2.imshow("Image", image)
cv2.waitKey(0)

# save the image 
if args.output is not None:
    output_filename = args.output
else:
    output_filename = "newimage0.jpg"
cv2.imwrite(output_filename, image)

# Optional: test the loss if jpg format 
if args.loop is not None:
    # write a loop to test jpg read and save identity
    prev_image = image
    diffs = []
    for i in range(int(args.loop)):
        if i == 0:
            filename = output_filename
        else:
            filename = "newimage" + str(i) + ".jpg"
        newfilename = "newimage" + str(i+1) + ".jpg"
        print("old file: {}, new file {}".format(filename, newfilename))
        image = cv2.imread(filename)
        diff_image = image-prev_image
        diff = np.abs(diff_image).sum()
        diffs.append(diff)
        print("loop {:02d}: {}".format(i, diff))
        # save the image 
        cv2.imwrite(newfilename, image)
        prev_image = image
        cv2.imshow("Iteration: {}".format(i), diff_image)
        cv2.waitKey(0)

    print(diffs)
    plt.semilogy(diffs)
    plt.title("sum of log diff between consecutive load/save")
    plt.xlabel("iteration #")
    plt.show()

    # remove temporary files
    for filename in glob.glob('newimage*.jpg'):
        os.remove(filename)
