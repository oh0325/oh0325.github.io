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

정리 1편 : ['[Python] 10가지 Python Tips!-(1)'](https://zzu0203.github.io/python/python-tips/)

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

