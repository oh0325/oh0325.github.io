---
layout: single
title: '[Python] if __name__=="__main__":의 역할!'
categories:
  - Python
tags:
  - python
comments: true  
classes: wide
toc: false
---
## | `if __name__=="__main__" :` 의 역할

파이썬 코드를 보면
```python
if __name__=="__main__":
  '''
  {코드}
  '''
```
처럼 `if __name__=="__main__" :`이 쓰여있는 코드가 많은데 이 역할이 정확히 무엇을 하는지에 대해 알아보도록 하겠습니다.

파이썬은 인터프리터에서 스크립트를 실행하게 된다면 함수나 클래스를 제외한 들여 쓰기 레벨 0인 모든 코드가 실행됩니다.

즉, 들여 쓰기가 되어있지 않은 모든 코드들이 실행됩니다.

다른 언어들과 다르게 파이썬은 `main()`함수가 자동으로 실행되지 않고 최상위 레벨에 있는 코드들을 `main()`함수로 암시적으로 생각합니다.

```python
def func():
  '''
  {코드}
  '''

if __name__=="__main__":
  '''
  {코드}
  '''
```
위 코드의 경우엔 `if`문 블록은 최상위 코드로 암시됩니다.

따라서 위의 스크립트를 인터프리터에서 실행시킨다면 `if` 블록 내의 코드들은 실행되지만 스크립트를 다른 모듈로 가져오는 경우에는 `if` 블록이 최상위 레벨이 아니기에 실행되지 않습니다.

`if __name__=="__main__" :`를 이용해 스크립트가 `import` 되어 실행되는지 직접 실행되는지를 판단할 수도 있습니다.

### | 예시

다음과 같은 file A.py와 B.py가 있다고 가정하겠습니다.
##### A.py
```python
def func():
    print('A에서 import 된 함수')

print('A의 최상위 레벨 코드')

if __name__=="__main__":
  print('A.py 스크립트를 직접 실행하였습니다.')

```

##### B.py
```python
import A

print('B의 최상위 레벨 코드')
A.func()

if __name__=="__main__":
  print('B.py 스크립트를 직접 실행하였습니다.')

```
인터프리터에서 위의 두 스크립트를 실행시킨 결과입니다.
```
>>>python A.py
A의 최상위 레벨 코드
A.py 스크립트를 직접 실행하였습니다.

>>>python B.py
A의 최상위 레벨 코드
B의 최상위 레벨 코드
A에서 import 된 함수
B.py 스크립트를 직접 실행하였습니다.

```
---
