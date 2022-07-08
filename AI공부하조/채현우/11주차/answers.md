# 11주차 합성곱 신경망
## 📒 이론문제 1
**다음 내용 중 틀린 설명을 모두 고르시오.**
**답 : 1, 4**

	1. CNN의 대표적인 모델은 VGGNet, AlexNet, GoogLeNet, ResNet 순으로 개발되었다.
	2. VGGNet은 layer를 깊게 만들게 하기 위해 3x3 크기의 커널을 사용했다.
	3. AlexNet은 하드웨어의 제약을 해결하기 위해 모델을 병렬 구조로 만들어 여러 gpu를 동시에 이용하도록 설계되었다.
	4. GoogLeNet은 VGG19보다 적은 층으로 구성되어 있고 1x1 크기의 커널을 이용해 특성 개수를 줄여 VGG보다 성능이 더 좋다.
	5. ResNet은 기존의 CNN 모델들에 비해 훨씬 깊은 층을 가지고 있고 각 노드 사이의 연산 방법을 수정하여 깊은 층을 거쳐도 모델의 학습이 잘 이루어 질 수 있도록 한다.

> 1.AlexNet, VGGNet 순으로 개발되었다.
> 
> 4.GoogLeNet은 22개의 층으로 구성, VGG19는 19층으로 형성되어 더 많은 층으로 구성된다.

## 📒 이론 문제 2
**다음 중 합성곱 신경망에 대해 잘못 설명한 것을 모두 고르세요.**
**답 : 1, 5**

	1. convolutional neural network에서 filter와 kernel은 같은 말로 사용하지만 뉴런은 다른 의미로 사용된다.
	2. 완전 연결 층만 사용하여 만든 신경망을 밀집 신경망이라고 부른다.
	3. 합성곱 계산을 통해 얻은 출력을 feature map이라고 부른다.
	4. 합성곱 층을 1개 이상 사용한 인공 신경망을 합성곱 신경망이라고 한다.
	5. 합성곱에서는 활성화 출력이란 표현을 잘 쓴다.

> 1.합성곱 신경망에서는 완전연결신경망과 달리 뉴런을 필터라고 부릅니다. 혹은 커널이라고도 부릅니다.
>
> 5.일반적으로 특성맵은 활성화 함수를 통과한 값을 나타냅니다. 합성곱에서는 활성화 출력이란 표현을 잘 쓰지 않습니다.

## ⚔ 실습 문제 3

**답안**

[Google Colaboratory](https://colab.research.google.com/drive/18dQ4m8tD4CakD38YtQ1Dy_QK5MrDn8Cs#scrollTo=TtQZ_BY3-8ib)



## 📒 이론 문제 4
**다음 중 CNN에 대해 틀린 것을 모두 고르시오.**
**답 : 3, 5**

	1. CNN은 필터링 기법을 인공신경망에 적용하여 이미지를 효과적으로 처리할 수 있는 심층 신경망 기법이다.
	2. CNN은 convolution layer과 pooling layer과 fully connected layer로 구성된다.
	3. CNN은 이미지 공간 정보 유실로 인한 정보 부족으로 인공 신경망의 학습이 정확도를 높이는데 한계가 있다.
	4. CNN은 이미지의 특징을 추출하는 부분과 클래스를 분류하는 부분으로 나눌 수 있다.
	5. CNN의 특징 추출 영역은 convolution layer과 pooling layer과 flatten layer로 구성된다.

> 1.flatten layer은 특징 추출 영역으로 구성되지 않는다.
> 
> 3.CNN은 공간 정보의 유실을 막기 위해 생겨난 알고리즘 입니다

## ⚔ 실습 문제 5

**답안**

[Google Colaboratory](https://colab.research.google.com/drive/18dQ4m8tD4CakD38YtQ1Dy_QK5MrDn8Cs#scrollTo=GYYsSPKVg3Xf)