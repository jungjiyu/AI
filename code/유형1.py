"""
유형1: Basic model (지도학습을 통하여 회귀모델만들기)
    :주어진 xs,ys 데이터 셋을 활용하여 기본적인 딥러닝과정을 거치면 된다.

특:
    :전처리 부분은 문제마다 다른거 주의
    :레이어가 인풋레이어 하나로 존나 단순, epoc 는 존나 큼
"""


#1:import
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Seqential


#2:전처리
'''
구글에서 준다
'''
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)



#3:모델링:모델정의
model = Seqential([
    Dense(1,input_shape[1]),
])


#4:컴파일-->모델생성
model.compile(optimizer = 'sgd' , loss = 'mse')


#5:학습
model.fit(xs,ys,
          epochs=1200,
          verbose = 1)

#6:예측-->predict([])
model.predict([10.0])
