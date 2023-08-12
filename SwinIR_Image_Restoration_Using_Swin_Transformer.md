# SwinIR: Image Restoration Using Swin Transformer

[Link to the article](https://arxiv.org/abs/2108.10257)

## Shallow Feature Extraction
The shallow feature extraction module of SwinIR is responsible for extracting low-frequency information from the input image. This module uses a convolutional layer to extract shallow features, which are then directly transmitted to the reconstruction module to preserve low-frequency information. The shallow feature mainly contains low-frequencies, which are important for preserving the overall structure and content of the image. By extracting shallow features separately, SwinIR can better preserve the low-frequency information and avoid losing important details during the restoration process.

In the case of SwinIR, the convolutional layer is used to extract low-frequency information from the input image, which is then directly transmitted to the reconstruction module to preserve this information. The convolutional layer is good at early visual processing, leading to more stable optimization and better results. It also provides a simple way to map the input image space to a higher dimensional feature space. By using a convolutional layer in the shallow feature extraction module, SwinIR can better preserve the low-frequency information and avoid losing important details during the restoration process.

## Deep Feature Extraction
the deep feature extraction module of SwinIR is responsible for capturing high-frequency information and long-range dependencies in the input image. The deep feature extraction module is composed of residual Swin Transformer blocks (RSTB) and a 3x3 convolutional layer. The RSTB is a building block that consists of several Swin Transformer layers and a residual connection. Each Swin Transformer layer in the RSTB utilizes local attention and cross-window interaction to capture complex patterns and dependencies in the input image. The residual connection in the RSTB allows the module to capture the residual information between the input and output of the block, which helps to improve the performance of the module. The deep feature extraction module is able to capture high-frequency information and long-range dependencies in the input image by utilizing the RSTB and the convolutional layer.

## Image Reconstruction
The image reconstruction part of SwinIR is responsible for generating a high-quality image from the extracted features. The image reconstruction part consists of two modules: the shallow feature extraction module and the deep feature extraction module. After the shallow and deep feature extraction modules, the high-quality image reconstruction module is used to generate the final output image. This module takes the extracted features from the previous modules and uses them to reconstruct a high-quality image. The high-quality image reconstruction module uses a convolutional layer for feature enhancement and a residual connection to provide a shortcut for feature aggregation. By combining the low-frequency information from the shallow feature extraction module and the high-frequency information from the deep feature extraction module, SwinIR is able to generate high-quality images with improved details and reduced artifacts.

The shallow feature F0 and the deep feature FDF are concatenated and passed through the reconstruction module, which is responsible for producing the final high-quality image. The reconstruction module is specific to each image restoration task, such as super-resolution or denoising.

## Residual Swin Transformer Block
The Residual Swin Transformer Block (RSTB) in the deep feature extraction module of SwinIR helps to find good features by utilizing several Swin Transformer layers and a residual connection. The Swin Transformer layers in the RSTB utilize local attention and cross-window interaction to capture complex patterns and dependencies in the input image. This allows the RSTB to extract high-frequency information and long-range dependencies from the input image, which are important for generating high-quality images.

The residual connection in the RSTB allows the module to capture the residual information between the input and output of the block. This residual connection helps to improve the performance of the module by allowing it to capture the difference between the input and output features. By capturing this residual information, the RSTB is able to learn more effective representations of the input image, which can lead to better performance in image restoration tasks.

## Experiments
* Classical Image Super-Resolution
* Image Denoising
* JPEG Compression Artifact Reduction

## Official Implementation
-  https://github.com/JingyunLiang/SwinIR




