---
layout: single
title: "WGAN - tensorflow/keras implementation"
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
date:   2020-07-24 00:00:00 
lastmod : 2020-07-27 20:30:00
sitemap :
  changefreq : daily
  priority : 1.0
---
### 논문 [WGAN](https://arxiv.org/abs/1701.07875)에 대한 tensorflow 코드 구현 입니다.

구현은 논문 8 페이지에 있는 아래의 **Algorithm 1**을 참고하였습니다.

![wgan algorithm](/assets/images/wgan_algo.PNG)

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

이렇게 얻어진 train_images를 tensorflow에서 지원하는 tf.data.Dataset을 사용해 batch 별로 Dataset object를 만들어줍니다.

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
summary한 결과는 다음과 같이 나옵니다. Generator는 약 228만개의 parameter, Discriminator는 약 1080만개의 parameter를 갖습니다. 

![model_g summary results](/assets/images/g_summary.PNG){: width="48%"}{: .center} ![model_d summary results](/assets/images/d_summary.PNG){: width="48%"}{: .center}

---
### Optimizer - RMSProp
논문의 알고리즘에 따르면 네트워크의 weights를 업데이트할 때 Optimizer로 RMSProp을 사용합니다. 따라서 다음과 같이 코드를 적어줍니다. 

```python
generator_optimizer = keras.optimizers.RMSprop(learning_rate)
discriminator_optimizer = keras.optimizers.RMSprop(learning_rate)
```
[케라스 홈페이지](https://keras.io/ko/optimizers/)에 따르면 RMSProp optimizer는 lr 기본값으로 0.001을 가집니다. 하지만, 알고리즘에 따라 lr 값을 0.00005로 적용시켜줍니다.

---
### Loss Functions
논문에 따라 ${x^{(i)}}$는 real data distribution으로부터 온 data sample이며 ${z^{(i)}}$는 latent vector입니다. 따라서 $f_{w}(x^{(i)})$는 critic(=discriminator)에 real sample을 넣은 결과값, $f_{w}(g_{\theta}(z^{(i)}))$는 critic에 generator가 만들어낸 fake sample을 넣은 결과값이 됩니다.

따라서 loss function에 대한 코드는 다음과 같습니다. d_loss는 real_output에 대한 mean 값과 fake_output에 대한 mean 값의 차이이며 g_loss는 fake_output에 대한 mean 값이 됩니다. 각 loss function은 functions.py 에 정의되어 있습니다.

```python
def d_loss(real_output, fake_output):
    loss = K.mean(real_output) - K.mean(fake_output)
    
    return loss

def g_loss(fake_output):
    loss = K.mean(fake_output)

    return loss
```

---
### seed 고정과 결과 이미지 생성
처음에 결과 이미지를 확인하기 위해 변수 num_examples_to_generate를 16(4*4)로 정의해두었습니다. 고정된 seed에 대해 결과 이미지가 변해가는 과정을 보기 위해서 다음과 같이 seed를 만들어줍니다.

```python
seed = tf.random.normal([num_examples_to_generate, noise_dim])
```
이렇게 만들어 놓은 seed를 사용해 결과화면에 4X4 형태로 보여주고 저장을 하려고 합니다.

```python
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('results/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
```
코드는 다음과 같으며 tensorflow 공식 홈페이지 내에 예제 코드입니다. 사용하다보니 편리하고 익숙해져서 예제코드의 큰 틀에 벗어나지 않게 코드 작성을 하였습니다.

mnist dataset처럼 흑백이 아닌 컬러 이미지를 dataset으로 사용한다면 `predictions[i, :, :, 0]` 부분의 0을  `predictions[i, :, :, :]` 다음과 같이 :로 바꿔주면 됩니다.

그리고 `cmap='gray'` 인자를 지워주거나 원하는 colormap 값으로 적으면 컬러 이미지 결과를 plot 할 수 있습니다. 
그리고 `plt.savefig('results/image_at_epoch_{:04d}.png'.format(epoch))`는 결과 영상을 저장하는 코드입니다.

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
checkpoint를 저장 할 directory를 변수 checkpoint_dir에 적어주면 됩니다. 

##### 추후 checkpoint save, restore 그리고 tensorboard 사용 등에 대해서도 자세히 다루도록 하겠습니다!!

---
### Train step(batch) function
이전까지 hyper-parameter setting, data pipelining, loss function, optimizer selecting 등에 대한 코드에 대해 설명을 했습니다. 드디어 **학습**과 관련한 코드입니다! 

다음의 코드는 논문의 알고리즘 내에 한 batch step에 대해서 critic(discriminator)과 generator의 loss를 구하고 parameter update하는 코드입니다. 우선 critic을 $n_{critic}=5$번 학습을 하고 generator를 학습하는 것이 한 step이 됩니다. critic의 loop안에서는 RMSProp optimizer를 통해 weights를 업데이트한 이후에 WGAN의 큰 특징 중 하나인 weight clipping을 합니다. weight clipping은  1-Lipschitz constraint를 강제하기 위해 수행됩니다.

코드는 다음과 같습니다.
```python
@tf.function
def train_step(images):
    for i in range(n): # n_critic 번 critic 학습
        noise = tf.random.normal([batch_size, noise_dim])
        with tf.GradientTape() as disc_tape:    # tf.GradientTape()을 이용해 gradient 계산
            D.training = True
            
            generated_images = G(noise)         # G로부터 fake data 생성
            real_output = D(images)             # 논문 내 f(x^i)
            fake_output = D(generated_images)   # 논문 내 f(G(z^i))
            disc_loss = d_loss(real_output, fake_output) # loss 계산
        
        # RMSProp(lr = 0.00005)로 학습 진행
        gradients_of_discriminator = disc_tape.gradient(disc_loss, D.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))
        
        # weight clipping
        disc_weights = discriminator_optimizer.weights  # get critic weights
        clip_w = [w.assign(tf.clip_by_value(w, -c, c)) for w in disc_weights if w.shape != ()]    # tf.clip_by_value를 통해 [-0.01, 0.01]로 clipping
    
    # generator 학습
    with tf.GradientTape() as gen_tape:
        noise = tf.random.normal([batch_size, noise_dim])
        G.training = True

        generated_images = G(noise)             # G로부터 fake data 생성     
        fake_output = D(generated_images)       # 논문 내 f(G(z^i))
        gen_loss = g_loss(fake_output)          # loss 계산
    
    # RMSProp(lr = 0.00005)로 학습 진행
    gradients_of_generator = gen_tape.gradient(gen_loss, G.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))    
    
    return gen_loss, disc_loss
```
train_step은 for문을 통해 critic이 $n_{critic}$번 학습을 먼저하게 됩니다. `discriminator_optimizer.apply_gradients`를 통해 gradient가 update되면 weight clipping을 합니다. 

weight clipping은 우선 `discriminator_optimizer.weights`처럼 optimizer의 weights method나 variables() method를 통해 얻을 수 있습니다. 제일 첫번째 tensor variable은 학습이 몇번째 iteration에 있는지 나타내는 tensor로 shape이 ()입니다. 

따라서 list comprehension을 통해 작성한 코드 `clip_w = [w.assign(tf.clip_by_value(w, -c, c)) for w in disc_weights if w.shape != ()]`를 보면 for문 뒤 조건문 `if w.shape != ()`을 통해 trainable한 weights만 clipping하도록 하였습니다. 

[`tf.clip_by_value()`](https://www.tensorflow.org/api_docs/python/tf/clip_by_value)를 통해 clipping을 했으며 `w.assign()`을 통해 disc_weights를 직접 업데이트 해주었습니다. 

---
### Training 
train 함수에서는 dataset과 epochs 값을 입력으로 받아 정해놓은 epoch 값 만큼 학습을 진행합니다. batch마다 얻어진 loss 값을 list에 담고 전체 epoch에 대한 평균 loss를 출력합니다. 그리고 한 epoch이 끝나면 4x4 형태로 결과를 plot하고 저장합니다. 또한 주석처리된 if문 내의 K에 적절한 값을 넣어 K epochs 마다 checkpoint에 모델을 저장할 수 있습니다.

```python
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        
        gen_loss_list = []
        disc_loss_list = []
        
        for image_batch in train_dataset:
            loss = train_step(image_batch)
            gen_loss_list.append(loss[0])
            disc_loss_list.append(loss[1])
            
        # 이미지 생성 및 저장
        display.clear_output(wait=True)
        generate_and_save_images(G, epoch + 1, seed)
        
        # K epochs 지날 때마다 모델 저장
        #if (epoch + 1) % K == 0:
        #    checkpoint.save(file_prefix = checkpoint_prefix)
    
        # loss & 시간 출력
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print ('G_Loss is {}, D_Loss is {}'.format(sum(gen_loss_list)/len(gen_loss_list), 
                                                   sum(disc_loss_list)/len(disc_loss_list)))

    # 학습이 끝난 후 이미지 생성
    display.clear_output(wait=True)
    generate_and_save_images(G, epochs, seed)
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

![wgan result](/assets/images/wgan_results.gif)

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

---
GAN에 대한 Tensorflow 구현을 차근차근 올리도록하겠습니다. 구현에 이상이 있거나 궁금한 내용은 편하게 댓글 달아주세요. 감사합니다.