# pisguru
OpenCV + Machine Learning

Notes:
- Make sure to have the correct xml file content and path for cascade detector. 


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