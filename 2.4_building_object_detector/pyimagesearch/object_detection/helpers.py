import imutils
import cv2

def pyramid(image, scale=1.5, minSize=(30, 30)):
    """ 
    Generator providing a series of rescaled images 

    This is lighter weight than calculating rescaled images and 
    store them in a list. Generator is great at creating pipelines.
    """
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # resize
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid 
        yield image

def sliding_window(image, stepSize, windowSize):
    """ slide a window across an image """
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # this slicing might yield truncated window at the boundary
            # add external check when this function is used
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def crop_ct101_bb(image, bb, padding=10, dstSize=(32, 32)):
    (y, h, x, w) = bb
    (x, y) = max((x - padding, 0)), max(y - padding, 0)
    roi = image[y:(h + padding), x:(w + padding)]

    roi = cv2.resize(roi, dstSize, interpolation=cv2.INTER_AREA)

    return roi