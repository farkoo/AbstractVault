# Swin-Unet Unet-like Pure Transformer for Medical Image Segmentation

[Link to the article](https://arxiv.org/abs/2105.05537)

The purpose of Swin-Unet is to propose a new architecture for medical image segmentation tasks. Swin-Unet is based on the Swin Transformer network, which is a pure transformer-based architecture for image classification and segmentation tasks. Swin-Unet combines the Swin Transformer network with the U-Net architecture to capture both local and global features of the input image. 

Swin-Unet is composed of an encoder, bottleneck, decoder, and skip connections, as shown in Figure 1. The encoder, bottleneck, and decoder are all constructed based on the Swin Transformer block. The Swin Transformer block is a hierarchical Transformer architecture that uses shifted windows to capture global and long-range semantic information interaction. 

The encoder extracts features from the input image, the bottleneck further compresses the features, and the decoder generates the final segmentation map. The skip connections are used to combine the low-level and high-level features to improve the segmentation accuracy.

## Architecture overview

<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/Swin-Unet%20Architecture.png">
  <br>
  <em>Figure 1: Swin-Unet Architecture</em>
</p>

### 1. Encoder
The encoder in Swin-Unet is based on the Swin Transformer network, which is a pure transformer-based architecture for image classification and segmentation tasks. The Swin Transformer network uses a hierarchical architecture with shifted windows to extract context features from the input image. The Swin Transformer network also uses self-attention mechanisms to capture both local and global features of the input image.

### 2. Bottleneck
The rule of the bottleneck in the Swin Transformer network is to learn deep feature representations while keeping the feature dimension and resolution unchanged. The bottleneck is an important component of the Swin Transformer network because it allows the model to capture both local and global features while maintaining spatial information, which is important for medical image segmentation tasks. 

### 3. Decoder
The decoder in Swin-Unet is based on the U-Net architecture, which is a popular architecture for medical image segmentation tasks. The decoder in Swin-Unet uses a symmetric Swin Transformer-based architecture with a patch expanding layer to perform the up-sampling operation and restore the spatial resolution of the feature maps. The decoder also uses skip connections to fuse the multi-scale features from the encoder with the up-sampled features in the decoder.


## Swin Transformer block

<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/Swin%20Transformer%20Block.png">
  <br>
  <em>Figure 2: Swin Transformer Block</em>
</p>

The Swin Transformer block is a building block of the Swin Transformer network, which is a pure transformer-based architecture for image classification and segmentation tasks. The Swin Transformer block is constructed based on shifted windows and is composed of several layers, including **LayerNorm (LN) layer**, **multi-head self-attention (MSA) module**, **residual connection**, and a **two-layer MLP with GELU non-linearity**. 
* The **LayerNorm (LN) layer** is responsible for normalizing the input feature maps across the channel dimension. This process helps to reduce the internal covariate shift and to improve the stability and convergence of the network during training.

* The **multi-head self-attention (MSA)** module is responsible for capturing the global dependencies between the input feature maps. It takes as input a set of feature maps and outputs a set of attended feature maps, where each attended feature map is a weighted sum of the input feature maps. This process allows the network to capture long-range dependencies and to attend to different parts of the input feature maps.

* The **residual connection** is responsible for learning the residual features between the input and output of the Swin Transformer block. This process helps to improve the gradient flow and the overall performance of the network by allowing the network to learn the difference between the input and output features.

* The **two-layer MLP with GELU non-linearity** is responsible for capturing the local dependencies between the input feature maps. It takes as input the attended feature maps from the MSA module and applies two fully connected layers with GELU activation function to capture the non-linear relationships between the input and output of the Swin Transformer block. This process allows the network to capture more complex features and to learn non-linear relationships between the input and output of the block.

### Shifted Windows
The Swin Transformer block differs from the conventional MSA module in that it is constructed based on **shifted windows**. Shifted windows is a mechanism used in the Swin Transformer block, which is a hierarchical Transformer architecture used in Swin-Unet for medical image segmentation. The shifted windows mechanism used in the Swin Transformer block allows it to capture both local and global features. The input feature map is partitioned into non-overlapping windows, and the self-attention operation is performed only within each window. The windows are shifted by a certain stride to capture global and long-range dependencies. 

By using shifted windows, the Swin Transformer block can capture both local and global information more efficiently and effectively than the conventional multi-head self-attention (MSA) module used in traditional convolutional neural networks. This is because the shifted windows mechanism allows the Swin Transformer block to capture global and long-range dependencies more efficiently and effectively, while also capturing local features within each window. 


## Local and Global Features Diffusion
In the Swin Transformer block, local and global features are diffused through a process called multi-head self-attention. 

Multi-head self-attention involves computing multiple sets of query, key, and value vectors for each position in the input feature map. Each set of vectors is called a "head" of self-attention. The outputs of each head are concatenated and passed through a linear layer to produce the final output of the self-attention operation. 

By computing multiple heads of self-attention, the Swin Transformer block can capture different aspects of the input feature map, including both local and global features. The outputs of each head are then combined to produce a final output that contains information from all heads. 

<br>
* Patch Partition: In the Swin Transformer network, the input image is divided into non-overlapping patches of a fixed size. This process is called patch partitioning, and it is performed to reduce the computational complexity of the network and to enable the use of self-attention mechanisms on large images.

* Linear Embedding: After the patches are extracted, a linear embedding layer is applied to project the feature dimension into an arbitrary dimension represented as C. This process is performed to transform the patch tokens into a higher-dimensional space, which allows the network to capture more complex features.

* Patch Merging: The patch merging layer is responsible for downsampling and increasing the dimension of the feature maps. It takes as input a set of feature maps and outputs a smaller set of feature maps with a larger spatial resolution and a higher feature dimension. This process is performed to reduce the spatial resolution of the feature maps while increasing their feature dimension, which allows the network to capture more abstract features.

* Patch Expanding: The patch expanding layer is responsible for upsampling the feature maps. It takes as input a set of feature maps with a lower spatial resolution and a higher feature dimension and outputs a larger set of feature maps with a higher spatial resolution and a lower feature dimension. This process is performed to increase the spatial resolution of the feature maps while reducing their feature dimension, which allows the network to recover the spatial information lost during downsampling.

* Linear Projection: The linear projection layer is applied to the upsampled features to output the pixel-level segmentation predictions. It takes as input a set of feature maps with a lower spatial resolution and a lower feature dimension and outputs a set of feature maps with the same spatial resolution as the input image and a feature dimension equal to the number of classes to be segmented. This process is performed to map the extracted features to the output space, which allows the network to generate the final segmentation predictions.


## Official Implementation
https://github.com/HuCaoFighting/Swin-Unet
