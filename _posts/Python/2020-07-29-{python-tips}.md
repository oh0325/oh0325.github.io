---
layout: single
title: '[Python] 10가지 Python Tips!-(1)'
categories:
  - Python
tags:
  - python
comments: true  
use_math: true
classes: wide
toc: false
date:   2020-07-29 00:00:00 
lastmod : 2020-07-29 00:00:00
sitemap :
  changefreq : daily
  priority : 1.0
---
YouTube 영상중 10가지 python tips에 대한 영상이 있어 정리해보았습니다. 

REF : [10 Python Tips and Tricks For Writing Better Code](https://www.youtube.com/watch?v=C-gEQdGVXbk)
---
## #1 if else문을 간략히 하자. (Ternary Conditionals)

다음과 같이 조건문을 사용하는 일반적인 코드에서
```python
condition = True

if condition:
    x = 1
else:
    x = 0

print(x)
```
아래의 코드와 같이 ternary conditionals식을 사용하자.
```python
condition = True

x = 1 if condition else 0

print(x)
```
같은 결과를 갖지만 불필요한 코드를 줄일 수 있고, 읽고 이해하기 쉬워진다. 

## #2 큰 수를 사용할 때 '_'를 사용하자. (Underscore Placeholders)

큰 수를 사용할 때 자릿수를 하나하나 세가면서 읽어야한다. 
```python
num1 = 10000000000
num2 = 100000000

total = num1 + num2

print(total)
```
아래의 코드처럼 언더바 '_'를 추가하면 가독성을 보장할 수 있다.
```python
num1 = 10_000_000_000
num2 = 100_000_000

total = num1 + num2

print(total)
print(f'{total:,}')
```
또한, 출력을 볼 때에도 print함수에 포맷팅을 사용하여 가독성을 늘릴 수 있다.

## #3 context manager를 사용하자. (Context Managers)

다음과 같이 text.txt 파일을 불러와 읽는 코드가 있다. 이후에 txt 파일 내의 글을 공백에 따라 split하여 words에 저장하고 words의 수를 출력한다.
```python
f = open('test.txt', 'r')

file_contents = f.read()

f.close()

words = file_contents.split(' ')
word_count = len(words)
print(word_count)
```
이런 코드는 파일을 손수 open하고 사용한 후에 close해줘야 하는 번거로움이 있다.
아래와 같이 context manager를 사용한 코드는 이러한 번거로움을 줄여준다. 
```python
with open('test.txt', 'r') as f:
    file_contents = f.read()

words = file_contents.split(' ')
word_count = len(words)
print(word_count)
```
`with open(...) as ...:` 구문안에 필요한 내용을 넣어 사용하면 된다. context manager는 사용이 끝난 리소스의 release를 보장해준다. 
