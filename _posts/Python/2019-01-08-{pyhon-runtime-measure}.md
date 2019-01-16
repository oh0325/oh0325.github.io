---
layout: posts
title: '[Python] 실행시간 측정 방법'
excerpt_separator: "<!--more-->"
categories:
  - Python
tags:
  - jeykll
  - github
  - python
---
## Python 실행시간 측정 방법

파이썬 코드내에서 코드의 실행시간을 측정하는 방법에 대해 알아보겠습니다.

##### 우선 코드를 먼저 확인하겠습니다.
---
```python
import time

tic = time.time()
'''
{코드}
'''
toc = time.time()

print(toc - tic)
```
* `import time`으로 `time`모듈을 `import`
* `time.time()` : 1970년 1월 1일 자정 이후로 누적된 초를 float 단위로 반환하는 함수!
* `tic`이라는 변수에 시작시간을 저장
* `toc`에는 프로그램이 끝나는 시간을 저장
* `toc - tic`을 통해 실행시간을 측정  

```python
print('%f' %(toc - tic))
print('%.3f' %(toc - tic))
```  

* 출력시 원하는 형태에 맞춰서 시간출력 가능
