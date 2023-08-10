# CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification

CrossViT is a dual-branch vision transformer that is designed to learn multi-scale features for image classification tasks. The architecture consists of two branches: a global branch and a local branch. The global branch processes the entire image at once, while the local branch processes smaller patches of the image.

The global branch of CrossViT is based on the DeiT architecture, which is a popular vision transformer architecture. The local branch, on the other hand, is based on a modified version of the ResNet architecture, which is a popular convolutional neural network architecture. The local branch is designed to capture local features in the image, while the global branch is designed to capture global features.

The two branches of CrossViT are combined using a cross-attention mechanism, which allows the network to selectively attend to different parts of the input based on learned attention weights. The cross-attention mechanism is used to fuse the features learned by the global and local branches at multiple scales, which allows the network to capture multi-scale features.

The authors also propose a simple yet effective token fusion scheme based on cross-attention, which is linear in both computation and memory to combine features at different scales. They compare this scheme to three other simple heuristic approaches and show that cross-attention achieves the best accuracy while being efficient for multi-scale feature fusion.

These schemes are illustrated in Figure 3 and described as follows:

a) All-attention fusion: In this scheme, all tokens are bundled together without considering any characteristic of tokens. This means that all tokens, including the CLS token and the patch tokens, are treated equally and combined using self-attention.

b) Class token fusion: In this scheme, only the CLS tokens are fused, as they can be considered as a global representation of one branch. The patch tokens are not fused and are passed through the transformer encoder separately.

c) Pairwise fusion: In this scheme, tokens at the corresponding spatial locations are fused together, and the CLS tokens are fused separately. This means that the patch tokens from the global and local branches are combined pairwise, based on their spatial location in the image.

d) Cross-attention: In this scheme, the CLS token from one branch and the patch tokens from another branch are fused together using cross-attention. This means that the global and local branches are combined using a cross-attention mechanism, which allows the network to selectively attend to different parts of the input based on learned attention weights.
