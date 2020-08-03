---
layout: single
title: '[Python] 10가지 Python Tips!-(2)-ing'
categories:
  - Python
tags:
  - python
comments: true  
use_math: true
classes: wide
toc: false
date:   2020-08-03 00:00:00 
lastmod : 2020-08-03 00:00:00
sitemap :
  changefreq : daily
  priority : 1.0
---
YouTube 영상 중 10가지 python tips에 대한 내용이 있어 정리해보았습니다. 

REF : [10 Python Tips and Tricks For Writing Better Code](https://www.youtube.com/watch?v=C-gEQdGVXbk)

정리 1편 : [[Python] 10가지 Python Tips!-(1)](https://zzu0203.github.io/python/python-tips/)

---
## #6 Unpacking에 대해 (Unpacking)

아래의 코드는 **#5번 Zip**에서 작성한 코드이다. `for name, hero, universe in zip(names, heroes, universes):`의 처음 부분에 `name, hero, universe`라는 변수 3개를 `zip()`으로부터 받는다. 

```python
names = ['Peter Parker', 'Clark Kent', 'Wade Wilson', 'Bruce Wayne']
heroes = ['Spiderman', 'Superman', 'Deadpool', 'Batman']
universes = ['Marvel', 'DC', 'Marvel', 'DC']

for name, hero, universe in zip(names, heroes, universes):
    print(f'{name} is actually {hero} from {universe}')
```
사실 이 경우에서 `zip()`은 `(name, hero, universe)`라는 3개의 값을 가진 하나의 **tuple**을 **unpacking**해서 반환한 것이다. 하지만 우리가 하나의 tuple로 값을 반환받고 싶다면 아래의 코드처럼 작성하면 된다.

```python
names = ['Peter Parker', 'Clark Kent', 'Wade Wilson', 'Bruce Wayne']
heroes = ['Spiderman', 'Superman', 'Deadpool', 'Batman']
universes = ['Marvel', 'DC', 'Marvel', 'DC']

for value in zip(names, heroes, universes):
    print(value)
```

그럼 하나의 tuple로 결과가 나오는 것을 확인할 수 있다. 즉 unpacking을 통해 3개의 변수 값을 개별적으로 접근할 수도 있지만 unpacking을 하지 않는다면 하나의 tuple로 접근하게 만들 수 있다.

이제 어떻게 unpacking이 동작하는지 코드와 함께 알아보자. 

```python
#Normal
items = (1, 2)

print(items)
```
위의 코드는 일반적으로 tuple을 값으로 받은 경우이다.

```python
#Unpacking
a, b = (1, 2)

print(a)
print(b)
```
unpacking을 이용해 여러개의 변수로 값을 받을 수 있다. 

```python
a, b = (1, 2)

print(a)
# print(b)
```
unpacking을 사용하더라고 변수 b를 사용하지 않을 수도 있다. 그렇다면 위의 코드와 같이 주석이나 사용을 하지 않을 것이다. (지금의 경우에는 `print()`) 

하지만 이런식의 코드를 작성하면 IDE나 EDITOR에서 변수 b를 선언했지만 사용하지 않았다는 경고를 할 것이다. 

```python
a, _ = (1, 2)

print(a)
# print(b)
```

python에서 사용하고 싶지 않은 값이 있다면 변수명을 `Underscore(_)`로 사용하면 된다. 즉, 값을 무시하고 싶은 경우 사용하면 되며 다른 사람의 코드를 보다가 `Underscore(_)`변수를 보면 암시적으로 이 값은 사용하지 않는다고 생각하면 된다.

{% capture title_url %}

python에서 `Underscore(_)`가 어떤 상황에서 사용되는지 나중에 정리하고 링크를 남기도록 하겠습니다.

{% endcapture %}
<div class="notice--info">{{ title_url | markdownify }}</div>

```python
a, b, c = (1, 2)

print(a)
```
다음과 같이 unpacking하는 값보다 변수가 많다면 `ValueError: not enough values to unpack (expected 3, got 2)`이라는 error message가 뜰 것이다.

```python
a, b, c = (1, 2, 3, 4, 5)

print(a)
```
반대로 다음과 같이 unpacking하는 값이 변수보다 많다면 어떻게 될까?

정답은 `ValueError: too many values to unpack (expected 3)`이라는 error message가 뜬다.

그렇다면 a에는 1을 b에는 2를 c에는 나머지 값 '3, 4, 5'를 할당하고 싶다면 어떻게 해야할까? 

다행이도 `Asterisk(*)`를 통해 unpacking은 위의 문제를 해결할 수 있다.

```python
a, b, *c = (1, 2, 3, 4, 5)

print(a)
print(b)
print(c)
```
결과로 다음과 같이 c에 나머지 값들이 할당되는 것을 확인할 수 있다.

```
결과 : 
1
2
[3, 4, 5]
```

만약 a, b를 제외한 c를 사용하고 싶지 않다면 마찬가지로 `Underscore(_)`를 사용해주면 된다.

```python
a, b, *_ = (1, 2, 3, 4, 5)

print(a)
print(b)
# print(c)
```
`Asterisk(*)`와 `Underscore(_)`를 통해서 사용하지 않을 여러개의 변수를 unpacking하는 방법을 알아보았다. 


```python
a, b, *c, d = (1, 2, 3, 4, 5)

print(a)
print(b)
print(c)
print(d)
```
자주 사용하지는 않지만 위와 같은 방법의 unpacking도 가능하다.

결과는 다음과 같이 나온다.
```
결과 : 
1
2
[3, 4]
5
```
변수 c에 앞의 2개의 a, b 그리고 마지막 d를 제외한 나머지 값들이 들어가 있는 것을 볼 수 있다.

```python
a, b, *c, d = (1, 2, 3, 4, 5, 6, 7)

print(a)
print(b)
print(c)
print(d)
```
unpacking할 tuple에 값을 더 추가해도 결과는 마찬가지로 변수 앞의 2개 그리고 마지막 d를 제외한 나머지 값들이 들어가 있는 것을 볼 수 있다.

```
결과 : 
1
2
[3, 4, 5, 6]
7
```

물론 가운데 c에 해당하는 값을 사용하고 싶지 않다면 `Underscore(_)`를 사용하면 된다.
```python
a, b, *_, d = (1, 2, 3, 4, 5, 6, 7)

print(a)
print(b)
# print(c)
print(d)
```
물론 결과는 다음과 같이 가운데 [3, 4, 5, 6] 없이 처음 두개의 값과 마지막 값만 출력된다.

```
결과 : 
1
2
7
```
