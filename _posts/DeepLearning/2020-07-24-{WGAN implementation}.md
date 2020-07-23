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
use_math: true
classes: wide
toc: false
---
### 논문 [WGAN](https://arxiv.org/abs/1701.07875)에 대한 tensorflow 코드 구현 입니다.

구현은 논문 8 페이지에 있는 아래의 **Algorithm 1**을 참고하였습니다.

![wgan algorithm](/assets/images/wgan_algo.PNG)

이제 코드와 함께 설명을 하도록 하겠습니다.

---
### Module Import
tensorflow, keras, numpy 등 필요한 모듈을 불러옵니다.

```python
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
# import tensorflow.keras.preprocessing.image as prep 
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from models import Generator_mnist, Discriminator_mnist # models.py 에 있는 model(G, D) load
from functions import g_loss, d_loss            # functions.py 에 있는 loss functions load
```
---
### Set Parameter
알고리즘을 보면 parameter 값으로 learning rate ($\alpha$) = 0.00005, clipping parameter ($c$) = 0.01, batch size ($m$) = 64, $n$<sub>critic</sub> = 5를 사용했습니다.

따라서 같은 값으로 parameter들을 설정합니다.
```python
learning_rate = 0.00005 # alpha
c = 0.01                # clipping parameter
n = 5                   # n_critic
epochs = 50
batch_size = 64
noise_dim = 100
num_examples_to_generate = 16
BUFFER_SIZE = 60000     # mnist buffer size
```
---
### Data Load
데이터셋은 mnist dataset을 사용했으며 추후 다른 데이터 셋에 대해 실험한 결과를 추가하도록 하겠습니다. 

```python
(train_images, _), (_, _) = mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # image normalization [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(batch_size)
```
mnist dataset은 숫자 0~9까지 60000개의 (28, 28, 1) shape을 갖는 흑백 이미지입니다. WGAN을 학습할 때 train label, validation set은 필요하지 않으므로 _ 로 사용하지 않음을 표시해줍니다.

불러온 6만장의 train_images는 (60000, 28, 28)의 shape을 하고 있으므로 마지막에 채널을 추가하기 위해 (60000, 28, 28, 1)의 shape으로 reshape을 합니다.
Generator의 마지막 activation function을 tanh로 사용했기에 image의 값을 [-1, 1]로 normalization 해줍니다. 

이렇게 얻어진 train_images를 tensorflow에서 지원하는 tf.data.Dataset을 사용해 batch 별로 Dataset object를 만들어 줍니다.

---
### Model(G, D) Load & Summary
models.py에서 불러온 model들을 확인해봅시다. 각 model들은 클래스 형태로 network을 구성하였습니다. model에 적절한 size의 Input을 넣고 model을 summary 합니다.

```python
G = Generator_mnist()
D = Discriminator_mnist()

input1 = keras.Input(shape=(noise_dim)) # noise_dim = 100
input2 = keras.Input(shape=(28, 28, 1))

fakeout = G(input1)
realout = D(input2)

G.summary()
D.summary()
```
summary한 결과는 다음과 같이 나온다. Generator는 약 228만개의 parameter, Discriminator는 1080만개의 parameter를 갖는다. 

![model_g summary results](/assets/images/g_summary.PNG){: width="48%"}{: .center} ![model_d summary results](/assets/images/d_summary.PNG){: width="48%"}{: .center}

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
- WGAN 논문 - <https://arxiv.org/abs/1701.07875>
- 케라스 공식 홈페이지 - <https://keras.io/>
- 텐서플로우 공식 홈페이지 - <https://www.tensorflow.org/api_docs>
