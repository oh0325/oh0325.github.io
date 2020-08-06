---
layout: single
title: "WGAN-GP - Tensorflow/Keras Implementation"
categories:
  - DeepLearning
tags:
  - gan
  - wgan-gp
  - wgan
  - wasserstein gan
  - tensorflow
comments: true
use_math: true
classes: wide
toc: false
date:   2020-08-07 00:00:00 
lastmod : 2020-08-07 00:00:00
sitemap :
  changefreq : daily
  priority : 1.0
---
### 논문 [WGAN-GP](https://arxiv.org/abs/1704.00028)에 대한 tensorflow 코드 구현 입니다.

구현은 논문 4 페이지에 있는 아래의 **Algorithm 1**을 참고하였습니다.

![wgan-gp algorithm](/assets/images/wgan_gp_algo.PNG)

이제 코드와 함께 설명을 하도록 하겠습니다.
{% capture title_url %}

- 학습은 가상환경의 **jupyter notebook** 에서 진행했습니다!
- 포스트 하단에 **dependency**에 대한 내용이 있습니다!

{% endcapture %}
<div class="notice--info">{{ title_url | markdownify }}</div>

---
### Module Import
tensorflow, keras, numpy 등 필요한 모듈을 불러옵니다.

```python
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
#import tensorflow.keras.preprocessing.image as prep 
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from models import Generator_mnist, Discriminator_mnist
from data_load import get_npdata, get_data_list, load_celeba_to_np

from IPython import display
```
---
### Set Parameter
알고리즘을 보면 parameter 값으로 gradient penalty coefficient ($\lambda$) = 10, $n$<sub>critic</sub> = 5, learning rate ($\alpha$) = 0.0001, adam hyperparameters $\beta$<sub>1</sub>, $\beta$<sub>2</sub> = 0, 0.9 를 사용했습니다.
batch size ($m$) = 64로 이전 [wgan 구현 포스트](https://zzu0203.github.io/deeplearning/WGAN-implementation/)와 동일한 값으로 설정했습니다. 

따라서 같은 값으로 parameter들을 설정합니다.
```python

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


![model_g summary results](/assets/images/g_summary.PNG){: width="48%"}{: .center} ![model_d summary results](/assets/images/d_summary.PNG){: width="48%"}{: .center}

---
### Optimizer - RMSProp


```python

```


---
### Loss Functions


```python

```

---
### seed 고정과 결과 이미지 생성
처음에 결과 이미지를 확인하기 위해 변수 num_examples_to_generate를 16(4*4)로 정의해두었습니다. 고정된 seed에 대해 결과 이미지가 변해가는 과정을 보기 위해서 다음과 같이 seed를 만들어줍니다.

```python
seed = tf.random.normal([num_examples_to_generate, noise_dim])
```
이렇게 만들어 놓은 seed를 사용해 결과화면에 4X4 형태로 보여주고 저장을 하려고 합니다.

```python

```


---
### Checkpoint Setting
학습 중간중간 일정 epoch마다 모델을 저장하기 위해 checkpoint를 setting합니다.

```python
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 G=G,
                                 D=D)
```


---
### Train step(batch) function

```python

```


---
### Training 

```python
```
저는 학습을 가상환경의 jupyter notebook에서 진행했습니다. 

jupyter cell에서 
```python 
%%time
train(train_dataset, epochs)
```
다음 코드를 실행시키면 학습을 진행할 수 있습니다. 

---
### Results
다음은 WGAN이 만들어낸 mnist data 결과입니다. 
1~2 epoch이 지난 이후 서서히 숫자 형태가 나타나고 10 epochs 정도가 지난후엔 꽤 그럴싸한 숫자를 만들어냈습니다. 

![wgan result](/assets/images/wgan_gp_results.gif)

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
- WGAN-GP 논문 - <https://arxiv.org/abs/1704.00028>
- 케라스 공식 홈페이지 - <https://keras.io/>
- 텐서플로우 공식 홈페이지 - <https://www.tensorflow.org/api_docs>

---
GAN에 대한 Tensorflow 구현을 차근차근 올리도록하겠습니다. 구현에 이상이 있거나 궁금한 내용은 편하게 댓글 달아주세요. 감사합니다.