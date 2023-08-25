# Awesome Image Enhancement

| Year |	Proceeding	| Title | PDF | Code |
| :---: | :---: | :---: | :---: | :---: |
| 2022 | ICPR | DocEnTr: An End-to-End Document Image Enhancement Transformer | [Click Here](https://arxiv.org/pdf/2201.10252.pdf) | [Click Here](https://github.com/dali92002/DocEnTR)
| 2023 | MDPI | Unsupervised Low Light Image Enhancement Transformer Based on Dual Contrastive Learning | [Click Here](https://bmvc2022.mpi-inf.mpg.de/0373.pdf) | [Click Here](https://github.com/KaedeKK/UDCL-Transformer)
| 2023 | MDPI | Low-Light Image Enhancement by Combining Transformer and Convolutional Neural Network | [Click Here](https://www.mdpi.com/2227-7390/11/7/1657/pdf?version=1680159122) | [Click Here]
| 2022 | CVPR | SNR-aware Low-Light Image Enhancement | [Click Here](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf) | [Click Here](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance) |
| 2022 | BMVC | You Only Need 90K Parameters to Adapt Light: A Light Weight Transformer for Image Enhancement and Exposure Correction | [Click Here](https://arxiv.org/pdf/2205.14871) | [Click Here](https://github.com/cuiziteng/Illumination-Adaptive-Transformer) |
| 2021 | ICCV | STAR: A Structure-aware Lightweight Transformer for Real-time Image Enhancement | [Click Here](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_STAR_A_Structure-Aware_Lightweight_Transformer_for_Real-Time_Image_Enhancement_ICCV_2021_paper.pdf) | [Click Here](https://github.com/zzyfd/STAR-pytorch) |
| 2023 | Arxiv | Self-Reference Deep Adaptive Curve Estimation for Low-Light Image Enhancement | [Click Here](https://arxiv.org/pdf/2308.08197v2.pdf) | [Click Here](https://github.com/john-venti/self-dace) |
| 2023 | CVPR | Learning a Simple Low-light Image Enhancer from Paired Low-light Instances | [Click Here](https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.pdf) | [Click Here](https://github.com/zhenqifu/pairlie) |
| 2022 | CoRR | Low-Light Image and Video Enhancement: A Comprehensive Survey and Beyond | [Click Here](https://arxiv.org/pdf/2212.10772.pdf) | [Click Here] |
| 2021 | Springer | Benchmarking Low-Light Image Enhancement and Beyond | [Click Here](https://sci-hub.se/10.1007/s11263-020-01418-8) | [Click Here] |
| 2021 | IEEE Trans | Low-Light Image and Video Enhancement Using Deep Learning: A Survey | [Click Here](https://arxiv.org/pdf/2104.10729) | [Click Here] |
| 2022 | IEEE Trans | Underwater Image Enhancement via Minimal Color Loss and Locally Adaptive Contrast Enhancement | [Click Here](https://drive.google.com/file/d/1d8gLKDPoxISy6-oVc6bQJZ-sElmNDPbB/view) | [Click Here]() |
| 2022 | IEEE Trans(TPAMI ) | Learning Enriched Features for Fast Image Restoration and Enhancement | [Click Here](https://arxiv.org/pdf/2205.01649) | [Click Here](https://github.com/swz30/MIRNetv2) |
| 2020 | ECCV | Learning Enriched Features for Real Image Restoration and Enhancement | [Click Here](https://arxiv.org/pdf/2003.06792v2.pdf) | [Click Here](https://github.com/swz30/MIRNet) |
| 2020 | CVPR | Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement | [Click Here](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf) | [Click Here](https://github.com/Li-Chongyi/Zero-DCE) |
| 2021 | IEEE Trans | underwater image enhancement via medium transmission-guided multi-color space embedding | [Click Here](https://arxiv.org/pdf/2104.13015.pdf) | [Click Here](https://github.com/Li-Chongyi/Ucolor) |
| 2023 | CVPR | Burstormer: Burst Image Restoration and Enhancement Transformer | [Click Here](https://openaccess.thecvf.com/content/CVPR2023/papers/Dudhane_Burstormer_Burst_Image_Restoration_and_Enhancement_Transformer_CVPR_2023_paper.pdf) | [Click Here](https://github.com/akshaydudhane16/Burstormer) |
| 2023 | AAAI  | Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method | [Click Here](https://arxiv.org/pdf/2212.11548) | [Click Here](https://github.com/TaoWangzj/LLFormer) |
| 2022 | IEEE Trans | U-shape Transformer for Underwater Image Enhancement | [Click Here](https://arxiv.org/pdf/2111.11843) | [Click Here](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement) |
| 2022 | IEEE Trans | twin adversarial contrastive learning for underwater image enhancement and beyond | [Click Here](https://drive.google.com/file/d/1rAQVj1RamqHI0OefUTnAeqV6vNFdAm0S/view) | [Click Here](https://github.com/Jzy2017/TACL) |

|  |  |  | [Click Here]() | [Click Here]() |

## Learning a Simple Low-light Image Enhancer from Paired Low-light Instances - CVPR2023
The purpose of this paper is to introduce PairLIE, an unsupervised approach for low-light image enhancement that learns adaptive priors from low-light image pairs. The paper aims to improve contrast and restore details for images captured in low-light conditions.

PairLIE is an unsupervised approach for low-light image enhancement that learns adaptive priors from low-light image pairs. The method consists of two main steps:

1. Retinex Decomposition: The network is expected to generate the same clean images as the two inputs share the same image content. To achieve this, the network is imposed with the Retinex theory and makes the two reflectance components consistent.
2. Self-Supervised Mechanism: To assist the Retinex decomposition, inappropriate features in the raw image are removed with a simple self-supervised mechanism.

Datasets:
* SICE (part2)
*  LOL (training set)
  
<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/IE_PairLIE_arch.png" alt="PairLIE Architecture">
  <br>
  <em>PairLIE Architecture</em>
</p>

<p align="center">
  <img src="https://github.com/farkoo/AbstractVault/blob/master/IE_PairLIE_result.png" alt="PairLIE Architecture">
  <br>
  <em>PairLIE Results</em>
</p>
