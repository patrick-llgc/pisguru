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
- Image classification and image localization are two distinctive topics in computer vision.


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