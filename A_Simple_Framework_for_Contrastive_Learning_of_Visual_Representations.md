# A Simple Framework for Contrastive Learning of Visual Representations

[Link to the article](https://arxiv.org/abs/2002.05709)

## Purpose: 
The purpose of the simCLR framework is to learn representations that capture meaningful and useful features from the input data, which can be effectively utilized for downstream tasks such as classification.

## Phases:
### (1) Data Augmentation Pipeline:
In this phase, a data augmentation pipeline is employed to apply a series of random transformations to the input images. The methods include **random crop** (with resize and random flip), **random color distortion**, and **random Gaussian blur**. Despite the seemingly limited set of operations, these augmentations create diverse views of each image, contributing to the model's ability to learn more robust and generalizable features.

### (2) Neural Network Mapping:
The second phase focuses on mapping the augmented images to latent representations. The authors adopt a neural network base encoder denoted as `f(·)`, which extracts representation vectors from the augmented data examples. The ResNet architecture is commonly used as the base encoder. The output of the encoder, denoted as `h`, is passed through a projection head `g(.)` to obtain a lower-dimensional feature vector `z`. The projection head, comprising one or more fully connected layers with non-linear activation functions, reduces the dimensionality of the feature vector while preserving crucial information. The primary objective is to ensure that the learned representations are invariant to data augmentation.
<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/2.jpg">
  <be>
</p>

### (3) Contrastive Loss:
In the contrastive loss phase, the simCLR framework employs a contrastive loss function to encourage augmented views of the same image to be similar and augmented views of different images to be dissimilar. The similarity between representations is evaluated using cosine similarity between the representations of two augmented views. By training the network to maximize the contrastive loss, the learned representations capture meaningful and useful features from the input data.
<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/1.jpg" alt="Noise Contrastive Estimator (NCE) loss">
  <br>
  <em>Figure 1: Noise Contrastive Estimator (NCE) loss</em>
</p>

## Implementation Details:
For this specific implementation, a `ResNet-18` is utilized as the ConvNet backbone. It processes images of shape `(96, 96, 3)`, following regular STL-10 dimensions, and produces vector representations of size `512`. The projection head `g(.)` consists of two fully-connected layers, each containing `512` units, resulting in a final `64-dimensional` feature representation `z`.

## Pairing Strategy:
In the simCLR method, only two augmented images are created for each original image, forming positive pairs. Additionally, each augmented image is paired with all other augmented images as negative pairs, excluding the original images. This comprehensive pairing strategy allows the model to distinguish not only between similar and dissimilar views of the same image but also between different images.
<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/3.jpg">
  <be>
</p>
    
## Conclusion:
The simCLR framework presents an effective self-supervised learning approach in computer vision. By combining data augmentation, neural network mapping, and contrastive loss, simCLR successfully learns informative visual representations from large-scale unlabeled datasets. These learned representations can be leveraged for various downstream tasks such as image classification, even in scenarios with limited labeled data during training.
