# TransEM: Residual Swin-Transformer based regularized PET image reconstruction
[Link to the article]( https://arxiv.org/abs/2205.04204)

PET image reconstruction is the process of creating a 3D image of the distribution of a radiotracer in a patient's body using data acquired from a PET scanner. It is an ill-posed inverse problem because the measured data is incomplete and noisy, and there are many possible solutions that could explain the data. This makes it difficult to accurately reconstruct the underlying image, and regularization techniques are often used to constrain the solution space and improve the quality of the reconstructed image.

## TransEM Architecture
The overall architecture of the residual swin-transformer based regularizer (RSTR) consists of a residual block with a Swin Transformer Layer (STL) and two convolutional layers. The RSTR is used as a regularizer in the ML-EM iterative framework to reconstruct standard-dose images from low count sinograms.

<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/TransEM_architecture.png">
</p>

### Maximum Likelihood Expectation Maximization (ML-EM)
EM stands for Expectation-Maximization, which is an iterative algorithm used in PET image reconstruction. The EM algorithm is a model-based approach that models the physical properties of the imaging system and the statistical properties of the measured data. 
The algorithm iteratively estimates the image and the system parameters by alternating between two steps: the E-step and the M-step. In the E-step, the algorithm estimates the probability distribution of the hidden variables (i.e., the image) given the measured data and the current estimate of the system parameters. In the M-step, the algorithm updates the estimate of the system parameters based on the estimated probability distribution of the hidden variables. The algorithm repeats these two steps until convergence is achieved.
The EM algorithm is widely used in PET image reconstruction because it can handle the Poisson noise statistics of the measured data and can model the physical properties of the imaging system. However, the excessive noise propagation from the measurements is a major disadvantage of the ML solution.

### Residual swin-transformer regularizer (RSTR)
The proposed residual swin-transformer based regularizer (RSTR) improves upon current CNN-based PET image reconstruction methods by addressing the limitations of the localized receptive fields of CNN-based regularizers. The RSTR incorporates a convolution layer to extract shallow features, followed by a swin-transformer layer to extract deep features. The deep and shallow features are then fused using a residual operation and another convolution layer. This approach improves the quality of the reconstructed image by incorporating regularization into the iterative reconstruction framework, resulting in better performance in both qualitative and quantitative measures compared to state-of-the-art methods.

### Pixel to Pixel Fusion
pixel-to-pixel fusion operations are a key component of the proposed method for PET image reconstruction. The pixel-to-pixel fusion operation is used to combine the image estimate obtained from the EM step with the image estimate obtained from the RSTR block. The pixel-to-pixel fusion operation is a simple element-wise addition of the two image estimates, which results in a fused image estimate that combines the advantages of both methods. The EM step is good at handling the Poisson noise statistics of the measured data and can model the physical properties of the imaging system. The RSTR block, on the other hand, is good at improving the regularization of the reconstructed image by incorporating both shallow and deep features.

## Number of Blocks
The proposed method for PET image reconstruction is composed of n blocks, where each block contains EM for image updating, RSTR for regularization, and a pixel-to-pixel fusion operation. Due to the limitation of hardware and image size, the number of subsets chosen for the experiment is 6, so the number of unrolled blocks is multiples of six. In the experiment conducted in the paper, the number of unrolled blocks is 60, which achieves the best performance.

## Experiments

<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/TransEM_1.png">
</p>

<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/TransEM_2.png">
</p>

<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/TransEM_3.png">
</p>
