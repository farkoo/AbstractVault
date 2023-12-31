# ColonFormer: An Efficient Transformer-based Method for Colon Polyp Segmentation
[Link to the article](https://arxiv.org/abs/2205.08473)

Colon polyp segmentation is crucial for detecting and preventing colorectal cancer. These abnormal colon growths can turn cancerous if untreated. While colonoscopy effectively identifies and removes polyps, its time-consuming nature and dependence on skilled professionals necessitate the development of a precise computer-aided diagnosis system. This system would enhance colonoscopy efficiency and decrease the risk of colorectal cancer.

## ColonFormer Architecture
 * Encoder
 * Decoder
 * Refinement Module

### Encoder
The encoder in ColonFormer is a hierarchically structured lightweight Transformer that can learn multi-scale features. The encoder takes the input image and extracts features at multiple scales. Here are the sections of the encoder:

**1. Mix Transformer (MiT) Blocks** <br />

The encoder uses Mix Transformer (MiT) blocks, which are hierarchical Transformer blocks that can represent both high-resolution coarse and low-resolution fine features. The MiT blocks contain three main parts: **Multi-head Self-Attention (MHSA) layers**, **Feed Forward Network (FFN)**, and **Layer Norm**. The MHSA layers capture long-range dependencies between features, while the FFN applies non-linear transformations to the features. The Layer Norm normalizes the features to improve the stability of the training process.

**2. Overlapping Patch Merging** <br />

The duty of Overlapping Patch Merging in the encoder of ColonFormer is to divide the feature map into overlapping patches and merge them to ensure local continuity around those patches. This process helps to capture both local and global context information in the input image. The overlapping patches are merged with a kernel that has a stride smaller than the kernel size, which ensures that the local continuity around the patches is preserved.


These multi-level features are then passed to the decoder for further processing. 

### Decoder 
The decoder in ColonFormer is designed to gradually fuse the prior global map produced by the Pyramid Pooling Module (PPM) with multi-scale feature maps yielded by the MiT backbone. The decoder is based on the UPer Decoder architecture, which is inspired by the U-Net architecture and has been shown to be effective for semantic segmentation tasks.

**1. Pyramid Pooling Module (PPM)** <br/>

The PPM is used to extract multi-scale features from the final block of the encoder. It produces a pyramid of pooling layers that simultaneously produce multi-scale outputs of the input feature map. The resulting feature maps, which form a hierarchy of features containing information at different scales and sub-regions, are then concatenated to produce an efficient prior global representation.

**2. UPer Decoder architecture** <br/>

The UPer Decoder architecture consists of a series of decoder blocks that gradually upsample the feature maps and fuse them with the prior global map produced by the PPM. Each decoder block consists of two main parts: a decoder block that upsamples the feature maps and a refinement module that refines the boundary of polyp objects in the global map for accurate segmentation. The refinement module uses a new skip connection technique to improve the accuracy of the segmentation.

**3. Convolutional Layers** <br/>

ColonFormer also uses convolutional layers to condense the information by emphasizing the coherence between neighboring elements and thus enhancing the resulting semantic map. This is important for accurately segmenting polyps in endoscopic images.


Overall, the decoder in ColonFormer is designed to capture global context information and fuse it with the multi-scale feature maps produced by the encoder to produce an accurate segmentation of polyps in endoscopic images.

### Refinement Module
The refinement module in ColonFormer is designed to gradually refine the boundary of the global map to yield the final accurate segmentation. The module is based on a deep supervision mechanism that uses the global map and two intermediate maps to train the network in a supervised manner.

The refinement module consists of a series of refinement blocks. Each refinement block consists of two main parts: a convolutional block that processes the feature maps and a refinement block that refines the boundary of the global map. The convolutional block uses a series of convolutional layers to process the feature maps and extract more informative features. The refinement block uses a series of convolutional layers and skip connections to refine the boundary of the global map.

In addition to the refinement blocks, the refinement module also uses a deep supervision mechanism to train the network in a supervised manner. The global map and two intermediate maps are passed into the training loss in a deep supervision manner. Before calculating the training loss, all refined maps are upsampled back to the original image input size.

The Refinement Module in ColonFormer consists of three modules: the Attention Refinement (AR) module, the Spatial Attention (SA) module, and the Channel Attention (CA) module. The order in which these modules are applied is as follows:

**1. Channel-wise Feature Pyramid (CFP) module** <br/>
The CFP module is used to extract features from the encoder in multi-scale views before the Refinement Module.

**2. Attention Refinement (AR) module** <br/>
The AR module is applied first to capture the global context information of the feature maps and refine the feature maps by recalibrating the channel-wise feature responses.

**3. Spatial Attention (SA) module** <br/>
The SA is applied next to capture the spatial dependencies between different regions of the feature maps and refine the feature maps by recalibrating the spatial feature responses.

**4. Channel Attention (CA) module** <br/>
The CA module is applied last to capture the inter-channel dependencies of the feature maps and refine the feature maps by recalibrating the channel-wise feature responses.

**5. Deep supervision** <br/>
Deep supervision is used during training with the Refinement Module to improve the training process. The Refinement Module produces three refined maps: the global map and two intermediate maps. These maps are passed into the training loss in a deep supervision manner, which means that the loss is calculated for each of the three maps separately.
   
## Loss Function
The loss function in ColonFormer is designed to optimize the network parameters and produce an accurate segmentation of polyps in endoscopic images. The loss function consists of two main parts: a weighted focal loss and a weighted IoU loss.

- The weighted focal loss is used to address the class imbalance problem in polyp segmentation. The class imbalance problem arises because polyps are rare objects in endoscopic images, and the majority of the image pixels belong to the background. The weighted focal loss assigns higher weights to the minority class (polyps) and lower weights to the majority class (background) to balance the contribution of each class to the loss function.

- The weighted IoU loss is used to measure the similarity between the predicted segmentation and the ground truth segmentation. The IoU (Intersection over Union) is a common metric used in image segmentation tasks to measure the overlap between the predicted segmentation and the ground truth segmentation. The weighted IoU loss assigns higher weights to the pixels that are more important for accurate segmentation and lower weights to the pixels that are less important.

The total loss of ColonFormer is calculated as the sum of the weighted focal loss and the weighted IoU loss.

## Experiments
The authors perform experiments on five popular datasets for polyp segmentation: Kvasir, CVC-Clinic DB, CVC-Colon DB, CVC-T, and ETIS-Larib Polyp DB. 

The experiments are divided into three parts:

**1. Comparison with State-of-the-Art Methods** <br/>
In this experiment, the authors compare the performance of ColonFormer with state-of-the-art CNN-based and Transformer-based networks using the same widely-used dataset configuration as suggested in [23]. The authors report the results in terms of mean Intersection over Union (mIoU) and Dice Similarity Coefficient (DSC) metrics.

**2. 5-Fold Cross-Validation** <br/>
In this experiment, the authors perform 5-fold cross-validation on the CVC-ClinicDB and Kvasir datasets. Each dataset is divided into five equal folds. Each run uses one fold for testing and four remaining folds for training. The authors report the results in terms of mIoU and DSC metrics.

**3. Experiment 3: Cross-Dataset Evaluation** <br/>
In this experiment, the authors perform cross-dataset evaluation with three training-testing configurations:

- CVC-ColonDB and ETIS-Larib for training, CVC-ClinicDB for testing
- CVC-ColonDB for training, CVC-ClinicDB for testing
- CVC-ClinicDB for training, ETIS-Larib for testing

The authors report the results in terms of mIoU and DSC metrics.

## Metrics
* mIoU stands for mean Intersection over Union, which is a measure of the overlap between the predicted segmentation and the ground truth segmentation. It is calculated as the average of the Intersection over Union (IoU) scores for all classes. IoU is the ratio of the intersection between the predicted segmentation and the ground truth segmentation to the union of the two.

* DSC stands for Dice Similarity Coefficient, which is another measure of the overlap between the predicted segmentation and the ground truth segmentation. It is calculated as twice the intersection between the predicted segmentation and the ground truth segmentation divided by the sum of the areas of the two.


Both mIoU and DSC range from 0 to 1, with higher values indicating better segmentation performance.

## Official Implementation
https://github.com/ducnt9907/ColonFormer
