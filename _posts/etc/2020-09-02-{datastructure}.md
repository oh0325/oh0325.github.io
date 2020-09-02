---
layout: single
title: 'Data 메모리 계산'
categories:
  - etc
tags:
  - data structure
comments: true  
classes: wide
toc: false
date:   2020-09-02 00:00:00 
lastmod : 2020-09-02 23:59:59
sitemap :
  changefreq : daily
  priority : 1.0
---
### 0. bit와 byte

컴퓨터는 On과 Off 두가지 상태만을 감지할 수 있고 이를 간단히 이진수 1, 0으로 표현한다. 이러한 두가지 상태를 표현하는 데이터 단위를 비트(bit)라고 하며 bit는 데이터 구성의 최소 단위이다.

가령 `10010101`이라고 이런 연속된 이진수가 8개 있다면 8 bit이라고 하며 또는 1 byte라고 말한다. 즉, `1 byte = 8 bit` 이다. 

{% capture title_url %}

  **더 큰 단위의 데이터 표현**
  - KB(Kilo Byte) : 1024 byte = 1024 * 8 bit
  - MB(Mega Byte) : 1024 KB
  - GB(Giga Byte) : 1024 MB
  - TB(Tera Byte) : 1024 GB

{% endcapture %}
<div class="notice--info">{{ title_url | markdownify }}</div>

### Data의 Memory 계산

GPU상에서 deep learning model들을 학습하다보면 OOM(Out Of Memory) error가 발생하는 경우가 많다. 보통 batch size를 줄이거나 model의 size를 줄이는 등의 방법으로 문제를 해결한다.

학습에 사용하는 batch size의 데이터들이 얼마나 메모리를 갖고 있는지 다음과 같이 확인한다.

Width * Height * 3(size of image channels) * batch_size * 32 bits를 계산하면 된다.

예를들어 `32*32*3`의 크기를 갖는 rgb image를 사용하고 batch size가 32라고 한다면 `32 * 32 * 3 * 32(batch size) * 32 bits` $\approx$ `3MB`

{% capture title_url %}

  **뒤에 32 bits를 곱하는 이유**
  data type을 float32라고 가정했을 때 float32 수 하나를 표현하기 위해 4 bytes의 메모리가 필요하다. 1 byte가 8 bits이므로 32bits를 곱해 계산한다. 

{% endcapture %}
<div class="notice--info">{{ title_url | markdownify }}</div>