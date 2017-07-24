import helpers

class ObjectDetector:
    def __init__(self, model, desc):
        # store the classifier and HOG descriptor
        self.model = model
        self.desc = desc

    def detect(self, image, winDim, winStep=4, pyramidScale=1.5, minProb=0.7):
        boxes = []
        probs = []

        # loop over the image pyramid
        for layer in helpers.pyramid(image, scale=pyramidScale, minSize=winDim):
            scale = image.shape[0] / float(layer.shape[0])

            # loop over the sliding window
            for (x, y, window) in helpers.sliding_window(layer, winStep, winDim):
                (winH, winW) = window.shape[:2]

                # ensure the window is not truncated 
                # since sliding_window does not check boundaries
                if winH == winDim[1] and winH == winDim[0]:
                    # extract HOG features from current window
                    # and classify if contains an object
                    features = self.desc.describe(window)

                    features = feature.hog(window, orientations=conf["orientations"], pixels_per_cell=tuple(conf["pixels_per_cell"]),
                        cells_per_block=tuple(conf["cells_per_block"]), transform_sqrt=conf["normalize"])

                    ## we need encapsulation