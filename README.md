# CAResNet

## A modified ResNet with channel attention mechanism

Code accompanying the paper "Efficient channel attention residual learning for the time-series fault diagnosis of wind turbine gearboxes" by Wenliao Du, Zhen Guo, Xiaoyun Gong, Yan Gu, Ziqiang Pu and Chuan Li (Ready to be submitted for publication).
-  Tensorflow 2.0 implementation
-  Inspired by Qilong Wang $et$ $al$. [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks] ([https://openaccess.thecvf.com/content_cvpr_2018/papers/Deshpande_Generative_Modeling_Using_CVPR_2018_paper.pdf](https://arxiv.org/pdf/1910.03151.pdf)), -  The efficient channel attention mechanism (ECA) gains more effect on feature extraction compared to traditional CNN.
-  ECA is combined with one-dimensional ResNet for the fault diagnosis of wind turbines.
-  This repository contains the reproduction of several experiments mentioned in the paper

## Requirements

- Python 3.6
- Tensorflow == 2.6.2
- Numpy == 1.19.2
- Keras == 2.6.0

Note: All experiments were executed with NVIDIA GeForce GTX 1650Ti

## File description
* `main-cnn`: One-dimensional CNN for fault classification.
* `main-eca`: One-dimensional CNN with ECA for fault classification.
* `main-res-eca`: One-dimensional ResNet with ECA for fault classification. (our proposed CAResNet)
* `main-resnet`: One-dimensional ResNet for fault classification.

## Implementation details
- The overall experiments include swd, wd, swd-sem and wd-sem are included in Run Main.ipynb. Directly using this file can get the results. Note that users should change the directory to successfully run this code.
- Hyperparameter settings: Adam optimizer is used with a learning rate of `2e-4` in both the generator and the discriminator; The batch size is `32`, and the total iteration is 10,000. LABDA (Weight of cycle consistency loss) is `10`. Random projection in SWD is `32`.

## Usage
The script with `.py` contains all the experiments (four scenarios: wd/swd/wd-sem/swd-sem).
Note: Due to the copyright, only part of the data set is uploaded. For more detail please contact Authors.

## Ackonwledgements
This work is supported in part by the National Natural Science Foundation of China (52175080), the Intelligent Manufacturing PHM Innovation Team Program (2018KCXTD029), and the National Nature Science Foundation of China (Grant No. 52275138).
