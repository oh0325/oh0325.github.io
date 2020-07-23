---
layout: single
title: "WGAN - tensorflow 구현"
categories:
  - DeepLearning
tags:
  - gan
  - wgan
  - wasserstein gan
  - tensorflow
comments: true
classes: wide
toc: false
---
### 논문 [WGAN](https://arxiv.org/abs/1701.07875)에 대한 tensorflow 코드 구현 입니다.

구현은 논문 8 페이지에 있는 아래의 **Algorithm 1**을 참고하였습니다.

![wgan algorithm](/assets/images/wgan_algo.PNG)

데이터셋은 mnist dataset을 사용했으며 추후 다른 데이터 셋에 대해 실험한 결과를 추가하도록 하겠습니다. 

알고리즘을 보면 parameter 값으로 learning rate ($\alpha$) = 0.00005, clipping parameter ($c$) = 0.01, batch size ($m$) = 64, $n$<sub>critic</sub> = 5를 사용했습니다.



---
### Dependencies
```
OS         : Ubuntu 18.04
GPU        : RTX2080ti
CUDA       : 10.0
CUDNN      : 7.6
-------------------------
python     : 3.7.4
tensorflow : 2.0.0-gpu
keras      : 2.2.4-tf
numpy      : 1.17.0
matplotlib : 3.1.1
```
---
### Reference
- WGAN 논문 - https://arxiv.org/abs/1701.07875
- 케라스 공식 홈페이지 - https://keras.io/
- 텐서플로우 공식 홈페이지 - https://www.tensorflow.org/api_docs
