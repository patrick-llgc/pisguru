# pisguru
OpenCV + Machine Learning

### Misc notes:

- Make sure to have the correct xml file content and path for cascade detector. 
- Points in openCV are defined in the order of (X, Y), but matrix is in the order of (nRow, nCOl).
- The best way to define points in openCV is in `tuple`. Convert `numpy.array.tolist()` to list and then to tuple before feeding into openCV functions. 
- The transformation matrix `cv2.warpAffine()` lists in the order of (X, Y). 
- To shrink an image, it will generally look best with cv2.INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with cv2.INTER_CUBIC (slow) or cv2.INTER_LINEAR (faster but still looks OK). Finally, as a general rule, cv2.INTER_LINEAR  interpolation method is recommended as the default for whenever you’re upsampling or downsampling — it simply provides the highest quality results at a modest computation cost.
- Flipping operations could be very useful in augment dataset (flipping images of faces to train machine learing algorithms)
- There is a difference between OpenCV and NumPy addition. NumPy will perform modulus arithmetic and “wrap around.” OpenCV, on the other hand, will perform clipping and ensure pixel values never fall outside the range [0, 255].
- `cv.add` behavior:
```
>>> cv2.add(np.uint8([10]), np.uint8([250]))
array([[255]], dtype=uint8)
>>> cv2.add(np.uint8([10]), np.uint8([260]))
array([[14]], dtype=uint8)
```
- Cropping can be realized through slicing numpy arrays, and also bitwise operations. Irregular cropping other than rectangular one can only be realized using bitwise operations.
- For simple masking, `cv2.bitwise_and(image, image, mask=mask)`
- `cv2.split` and `cv2.merge` are convenient functinos to convert between 3ch color and 1ch graysale images.
- **Global image classification** and **local object detection** are two distinctive topics in computer vision. The latter can be seen as a special case of the former (when the bouunding box or image size is sufficiently small).
- An object is what can be represented as a semi-grid structure with noticeable patterns. Objects in real world have substantial variations which makes classification and detection difficult (e.g., viewpoint, deformation, occulusion, illumination, background clutter, intra-class variation, etc). 
- template matching is the simplest form of object detection technique. The template must be nearly identical to the object to be detected for this technique to work.
- In template matching, the matching method is very critical. Generally `cv2.TM_CCOEFF` gives good results. 
- dlib is a cross platform computer vision library written in C++. It is best used to train object detection.
- Training an **object detector** is very similar to training a classifier to (globally) classify the contents of an image, however, both the labels of the images and **annotations**  corresponding to the bounding box surrounding each objectare needed. Features could be extracted from the bounding boxes, and then used to build our object detector.
- .mat files in Matlab are serialized, like .pickle files in python. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.
- kernels are used with convolution to detect edges
- **Smoothing** and **blurring** is one of the most common pre-processing steps in computer vision and image processing
- Applying Gaussian smoothing does *not always* reduce accuracy of object detection. It is generally suggested to run two experiemnts, one with and one without Gaussian filter and pick whichever gives better accuracy.  It is [shown](https://gurus.pyimagesearch.com/wp-content/uploads/2015/05/dalal_2005.pdf) that applying Gaussian filter at each layer of the pyramid can actually hurt the performance of the HOG descriptor — hence we skip this step in our image pyramid. 
- Use generator `yield` to create pipeline if there is too much data involved. 
- Choosing an appropriate **sliding window size** is critical to obtaining a high-accuracy object detector.



## Topics
### virtualenv in iPython notebook [link](https://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs/)
- Install the ipython kernel module into your virtualenv

```
workon my-virtualenv-name  # activate your virtualenv, if you haven't already
pip install ipykernel
```

- Now run the kernel "self-install" script:
```
python -m ipykernel install --user --name=my-virtualenv-name
```
Replacing the --name parameter as appropriate.

- You should now be able to see your kernel in the IPython notebook menu: Kernel -> Change kernel and be able so switch to it (you may need to refresh the page before it appears in the list). IPython will remember which kernel to use for that notebook from then on.



### key openCV functions
- `cv2.getRotationMatrix2D(center, angle, scale)`
- `cv2.warpAffine(src, M, dsize)`


### JPEG
A basic explanation can be found [here](http://www.whydomath.org/node/wavlets/basicjpg.html) and [here on stack exchange](https://photo.stackexchange.com/a/34264) and in this [ppt](http://www.dmi.unict.it/~battiato/EI_MOBILE0708/JPEG%20(Bruna).pdf), which includes 4 major steps. 

- Preprocessing: 
	- RGB-> YCbCr space; 2x subsampling of chroma components leads to 2x compression (1+1/4+1/4 = 1.5 = 3**/2**). 
	- Partitioning into 8x8 blocks; 
	- subtract 127
- Transformation: Discrete Cosine Transformation ([DCT](http://www.whydomath.org/node/wavlets/dct.html)); 
	> Loosely speaking, the DCT tends to push most of the high intensity information (larger values) in the 8 x 8 block to the upper left-hand of C with the remaining values in C taking on relatively small values. C = U B U^T. 

	> DCT is reversible. DCT coefficients can be viewed as weighting functions that,
when applied to the 64 cosine basis functions of various
spatial frequencies (8 x 8 templates), will reconstruct the
original block.
- Quantization. This step leads to [information loss](http://www.whydomath.org/node/wavlets/quantization.html). 
- Encoding: 
	- Zigzag ordering
	- [Huffman encoding](http://www.whydomath.org/node/wavlets/imagecompression.html)

### RGB vs BGR
- BGR is used in openCV for historical reasons.
- BGR images are stored in a [row-major order](http://docs.opencv.org/2.4/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#how-the-image-matrix-is-stored-in-the-memory)
![](http://docs.opencv.org/2.4/_images/math/b6df115410caafea291ceb011f19cc4a19ae6c2c.png)

### Kernels
- Kernel visualization [tool](http://setosa.io/ev/image-kernels/)
- Sobel filters are directional. Left Sobel filter is different from right Sobel filter, also both are veritical Sobel filters. When both edges need to be detected, it is better to keep data type to higher forms, take abosolute value and conver back to np.uint8 or cv2.cv_8U.

### Structuring Element and Morphological Operations
- Structuring elements are similar to kernels, but not used in the context of convolution.
- Intro to morphological operatinon [link](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
- The white top-hat transform returns an image, containing those "objects" or "elements" of an input image that:
	- Are "smaller" than the structuring element (i.e., places where the structuring element does not fit in), and
	- are brighter than their surroundings.
- The black top-hat (or black hat) returns an image, containing the "objects" or "elements" that:
	- Are "smaller" than the structuring element, and
	- are darker than their surroundings.
- In other words, top hat returns features that **would be eroded**; black hat returns features that **would be dilated**. 

### Blurring and Smoothing
- Median filter is nonlinear and best at removing "salt and pepper" noise, and creates "artistic" feeling as the image would appear blocky. Median filter will remove substantially more noise. Gaussian blur appears more natural.
- Bilateral filtering considers both the coordinate space and the color space (domain and range filtering, ergo **bi**lateral). Only if the neighboring pixel's value is close enough will it be contributing to the blurring of the pixel of interest. This will selectively blur the image and will not blur the edges. Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications, and perhaps d=9 for offline applications that need heavy noise filtering. (perhaps meitu used it for remove wrinkles!)
- In both Gaussian blur and bilateral filtering, there are two parameters, d and sigmaSpace that controls behavior in the coordinate space. It is the best practice to specify both. d is the finite (truncated) support of an infinite Gaussian surface with sigmaSpace.

### Lighting condition
-	The camera is not actually “filming” the object itself but rather the light reflected from our object. The success of (nearly) all computer vision systems and applications is determined before the developer writes a single line of code. Lighting can mean the difference between success and failure of your computer vision algorithm. The quality of light in a given environment is absolutely crucial, and perhaps the **most important** factor in obtaining your goals. You simply cannot compensate for poor lighting conditions. 
- The ideal lighting condition should yield high contrast in the ROI, be stable (with high repeatability) and be generalizable (not specific to a particular object).
- It is extremely important to at least consider the stability of the lighting conditions before creating a computer vision system.s

### color space
- RGB: Most popularly used. Additive color space, but color definition is not intuitive.
- HSV: 
	- HSV is much easier and more intuitive to **define a valid color** range using HSV than RGB.
	- The HSV color space is used heavily in computer vision applications — especially if we are interested in tracking the color of some object in an image. 
	- Hue: how pure the color is
	- Saturation: how white the color is. Saturation = 0 is white, while fully saturated color is pure.
	- Value: lightness, black to white
- L\*a\*b\*: 
	- The Euclidean distance between two arbitrary colors in the L\*a\*b\* color space has actual **perceptual meaning**.
	- It is very useful in color consistency across platforms, color management. 
	- The V(alue) channel in HSV is very similar to the L-channel in L\*a\*b\* and to the grayscale image.
- Grayscale: Grayscale is often used to save space when the color information is not used. Biologically, our eyes are more sensitive to green than red and then than blue. Thus when converting to grayscale, each RGB channel is not weighted uniformly:
$$Y = 0.299 \times R + 0.587 \times G + 0.114 \times B$$
Human beings perceive twice green than red, and twice red than blue.

### Thresholding
- To mask the object in a white background, use `cv2.THRESH_BINARY_INV`. 
- Otsu's method assumes a **bi-modal distribution** of grayscale pixel values. Otsu's method works best after Gaussian blur, which helps to make the histogram more bimodal. 
	- While using Otsu's method to find threshold, specify the threshold to 0 and supply additional `cv.THRESH_OTSU` to options: `cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU`
	![](http://docs.opencv.org/trunk/otsu.jpg)
	
### Gradients
- We use **gradients** for detecting **edges** in images, which allows us to find contours and outlines of objects in images. We use them as inputs for quantifying images through feature extraction — in fact, highly successful and well-known image descriptors such as **Histogram of Oriented Gradients (HOG)** and **SIFT** are built upon image gradient representations. Gradient images are even used to construct **saliency maps**, which highlight the subjects of an an image.
- Image with detected edges are generally called **edge maps**.
- Gradient magnitude and orientation make for excellent features and image descriptors when quantifying and abstractly representing an image. 
- **Scharr** kernel may give better approximation of the gradient than **Sobel** kernel.

	```
	Scharr = [[+3, +10, +3],
			  [0,    0,  0],
			  [-3, -10, -3]]
			  
	Scharr = [[+1, +2, +1],
			  [0,   0,  0],
			  [-1, -2, -1]]
	```
- The magnitude and orientation of the gradient:

	```
	mag = np.sqrt(gX ** 2 + gY ** 2)
	orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
	```

	Note that `gX` and `gY` should be in signed float. If we wish to visualize `gX`, we can use `cv2.convertScaleAbs(gX)` to convert signed float to unsigned 8-bit int. [openCV docs](http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html)

### HOG + Linear SVM object detector
- HOG + Linear SVM is superior to Haar Cascade (Viola-Jones Detector):
	- Haar cascades are extremely slow to train, taking days to work on even small datasets.
	- Haar cascades tend to have an **alarmingly high false-positive rate** (i.e. an object, such as a face, is detected in a location where the object does not exist).
	- What’s worse than falsely detecting an object in an image? Not detecting an object that actually does exist due to sub-optimal parameter choices.
	- Speaking of parameters: it can be especially challenging to tune, tweak, and dial in the optimal detection parameters; furthermore, the optimal parameters can vary on an image-to-image basis!
- HOG + Linear SVM entails 6 steps:
	1. Obtain positive examples by extracting HOG features from ROIs within positive images
	2. Obtain negative examples by extracting HOG features from negative images
	3. Train linear SVM.
	4. Obtain hard negative examples by collecting false positives on negative images
	5. Train linear SVM using hard negative mining (HNM) to reduce false positive rate.
	6. Non maximum suppression (NMS) to select only one bounding box in one neighborhood.
- It should be noted that the above steps could be simplified by obtaining positive AND negative examples from the ROI and other parts of the training positive images, respectively. This is implemented in `dlib` library.

### Canny Edge detection
- Canny edge detection entails four steps:
	1. Gaussian blur
	2. Sobel x and y gradients
	3. Non maximum suppression
	4. Hysteresis thresholding 
- The optimal value of the lower and upper boundaries of the **hysteresis thresholding** in Canny detection `cv2.Canny(image, lower, upper)` varies from image to image. In practice, setting upper to 1.33 * image medium and lower to 0.67 * image medium yields very good results. Remember the magic number **sigma=0.33**.


### Contours
- Contours can sometimes replace machine learning at solving some problems efficiently.
- For better contour extraction accuracy, it is preferable to use **binary** image rather than grayscale image.
- It is more productive to use `cv2.drawContours()` to draw all contours while slicing the contour list, rather than specifying the contour index, i.e., `cv2.drawContours(clone, cnts[:2], -1, (0, 255, 0), 2)`

