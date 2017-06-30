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


## Topics
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

