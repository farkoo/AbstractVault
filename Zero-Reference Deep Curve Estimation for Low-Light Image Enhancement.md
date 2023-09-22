# Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement

[Link to the article](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)

## Innovations:
1. The proposed low-light enhancement method is data-independent, reducing overfitting risk and performing well in various lighting conditions.
2. An image-specific curve is designed for versatile pixel-level mapping.
3. The model is trained without reference images, utilizing specialized loss functions to evaluate enhancement quality.

## Method:

<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/Images/Zero-DCE.png">
  <br>
  <em>Figure 1: Zero-DEC Framework</em>
</p>

A Deep Curve Estimation Network (DCE-Net) is devised to estimate a set of best-fitting Light-Enhancement curves (LE-curves) given an input image. The framework then maps all pixels of the input’s RGB channels by applying the curves iteratively for obtaining the final enhanced image.
The important components of this method are as follows:

### Light-Enhancement Curve (LE-curve)

1. Global Curve Estimation:

Zero-DCE aims to estimate a global enhancement curve or function that represents a transformation from the pixel intensities of the input image to their corresponding enhanced values. This global curve is learned by the DCE-Net, a deep neural network architecture.
The global curve captures the overall enhancement pattern to improve the visibility and quality of the entire image. It helps in adjusting the brightness, contrast, and other global attributes of the image.

2. Pixel-Wise Curve Estimation:

In addition to the global curve, Zero-DCE also focuses on pixel-wise curve estimation. This means that the enhancement process is applied individually to each pixel in the image.
The pixel-wise approach allows for fine-grained control over how different regions and details in the image are enhanced. It ensures that enhancements are adapted to the specific characteristics and lighting conditions of each pixel.
By estimating pixel-wise curves, Zero-DCE can enhance local details and features, ensuring that objects and details in the image are well-preserved and improved.

<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/Images/Zero-DCE%20LE.png">
  <br>
  <em>Figure 2: Light-Enhancement Curve </em>
</p>

### DCE-Net
To learn the mapping between an input image and its best-fitting curve parameter maps, we propose a Deep Curve Estimation Network (DCE-Net). The input to the DCE-Net is a low-light image while the outputs are a set of pixel-wise curve parameter maps for corresponding higherorder curves.

In the approach, a plain CNN is utilized, comprising seven convolutional layers with symmetrical concatenation. Each layer features 32 convolutional kernels of size 3×3 and stride 1, followed by the ReLU activation function. Down-sampling and batch normalization layers, which disrupt pixel relationships, are omitted. The final convolutional layer employs the Tanh activation function, generating 24 parameter maps for 8 iterations (n = 8). Each iteration necessitates three curve parameter maps for the three channels.

### Non-Reference Loss Functions
1. Spatial Consistency Loss:
This loss encourages the network to maintain spatial consistency in the enhanced image. It ensures that the relative positions of objects and details are preserved during enhancement.

2. Exposure Control Loss:
Exposure control loss helps in adjusting the brightness and exposure of the enhanced image, ensuring that it is visually pleasing and maintains appropriate lighting levels.

3. Color Constancy Loss:
Color constancy loss helps maintain consistent colors in the enhanced image, reducing color shifts that can occur during enhancement.

4. Illumination Smoothness Loss:
This loss encourages the illumination map estimated by the DCE-Net to be smooth and continuous, reducing artifacts and improving the overall quality of the enhanced image.

<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/Images/Zero-DCE%20loss.png">
  <br>
  <em>Figure 3: Total Loss </em>
</p>

## Official Implementation
https://github.com/Li-Chongyi/Zero-DCE
