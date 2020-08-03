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
lastmod : 2020-08-02 00:00:00
sitemap :
  changefreq : daily
  priority : 1.0
---
YouTube 영상 중 10가지 python tips에 대한 내용이 있어 정리해보았습니다. 

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
아래의 코드처럼 `Underscore(_)`를 추가하면 가독성을 보장할 수 있다.
```python
num1 = 10_000_000_000
num2 = 100_000_000

total = num1 + num2

print(total)
print(f'{total:,}')
```
또한, 출력을 볼 때에도 print함수에 포맷팅을 사용하여 가독성을 늘릴 수 있다.
{% capture title_url %}

`Underscore(_)`는 python3.6 이상의 버전에서 사용가능합니다.

{% endcapture %}
<div class="notice--info">{{ title_url | markdownify }}</div>


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
`with open(...) as ...:` 구문 안에 필요한 내용을 넣어 사용하면 된다. context manager는 사용이 끝난 리소스의 release를 보장해준다. 

## #4 Enumerate를 사용하자. (Enumerate)

python은 다음과 같이 간단하게 어떤것의 loop를 돌 수 있다. 지금은 names라는 list의 loop를 도는 것이다. 
```python
names =['Corey', 'Chris', 'Dave', 'Travis']

for name in names:
    print(name)
```
하지만 loop를 돌면서 index를 함께 count한다고 해보자. 많은 초보자의 경우 아래와 같은 코드를 생각할 것이다. 
```python
names =['Corey', 'Chris', 'Dave', 'Travis']

index = 0
for name in names:
    print(index, name)
    index += 1
```
작동은 잘 하지만 더 명료한 방법이 존재한다.
다음과 같이 enumerate function을 사용하는 것이다. 
```python
names =['Corey', 'Chris', 'Dave', 'Travis']

for index, name in enumerate(names):
    print(index, name)
```
enumerate function은 list의 index와 value를 둘 다 반환해준다. unpacking을 이용해 for문에 적용할 수 있다.
```python
names =['Corey', 'Chris', 'Dave', 'Travis']

for index, name in enumerate(names, start=1):
    print(index, name)
```
만약 시작 index를 0이 아닌 1로 시작하고 싶다면 enumerate function의 start argument를 1로 설정하면 된다.

{% capture title_url %}

index를 사용하고 싶지 않은 경우 underscore(_)를 이용해 `for _, value in list:`처럼 표현할 수 있습니다. 

{% endcapture %}
<div class="notice--info">{{ title_url | markdownify }}</div>

## #5 Zip을 사용하자. (Zip)

예를들어 두 개의 list의 원소들을 같이 봐야한다고 하자. enumerate를 사용한다면 loop의 index로 다른 list의 원소를 가르킬 수 있다. 
```python
names = ['Peter Parker', 'Clark Kent', 'Wade Wilson', 'Bruce Wayne']
heroes = ['Spiderman', 'Superman', 'Deadpool', 'Batman']

for index, name in enumerate(names):
    hero = heroes[index]
    print(f'{name} is actually {hero}')
```
위의 코드는 list `names`를 enumerate를 사용해 loop를 돈다. 이때 index를 통해 list `heroes`의 원소에 접근할 수 있다.

하지만 이런 방법은 직관적이지 않다. 동일한 코드에 대해 python에서는 zip function을 사용하는 것이 더 직관적이다.
```python
names = ['Peter Parker', 'Clark Kent', 'Wade Wilson', 'Bruce Wayne']
heroes = ['Spiderman', 'Superman', 'Deadpool', 'Batman']

for name, hero in zip(names, heroes):
    print(f'{name} is actually {hero}')
```
다음 코드는 enumerate 대신 zip을 사용한 코드로 `zip()`의 인자로 순회하고자 하는 list를 넣어주면 된다. zip function을 사용하며 위 코드와 같이 2개의 list를 한번에 순회할 수 있고 loop를 돌기 전에 unpacking해서 원소들을 묶어줄 수 있다. 이렇게 zip function을 사용하면 훨씬 직관적이고 깔끔한 코드를 작성할 수 있다.

또한 2개가 아닌 더 많은 리스트를 사용할 수도 있다. 밑의 코드는 list `universes`를 추가해 사용하였다.
```python
names = ['Peter Parker', 'Clark Kent', 'Wade Wilson', 'Bruce Wayne']
heroes = ['Spiderman', 'Superman', 'Deadpool', 'Batman']
universes = ['Marvel', 'DC', 'Marvel', 'DC']

for name, hero, universe in zip(names, heroes, universes):
    print(f'{name} is actually {hero} from {universe}')
```

{% capture title_url %}

하지만 zip function은 길이가 다른 list를 사용할 때 가장 길이가 짧은 list를 순회하고 멈춘다. 길이가 가장 긴 list의 끝까지 순회하고 싶다면 [`itertools.zip_longest()`](https://docs.python.org/3/library/itertools.html#itertools.zip_longest)를 사용하면 된다. 
- `zip_logest(*iterables, fillvalue=None)`의 인자 fillvalue는 순회를 다한 짧은 list에 대해 unpacking value를 default `None`으로 반환합니다. 원하는 value 값을 인자로 넣어주면 됩니다.

{% endcapture %}
<div class="notice--info">{{ title_url | markdownify }}</div>

---

다음 포스트 [10가지 Python Tips!-(2)]에서는 zip function에 대해 언급하면서 나온 unpacking에 대한 내용부터 정리하겠습니다.