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

