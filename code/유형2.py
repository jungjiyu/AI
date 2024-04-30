#유형2: 지도학습을 통한 분류기 만들기
"""
    : 분류하는 것은 둘중 하나다
        1. 이미지 : fashion, minisst
        2. 정형데이터 : iris
"""

#import

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Seqential
from tensorflow.keras.callbacks import ModelCheckpoint

#2전처리
'''
x데이터만 정규화 해주면 된다. x데이터만 이미지이기 때문.(y 데이터는 라벨이다)
fashion_mnist 을 두개의 튜플로 나누어 받는 이유는 fashion_mnist의 split 이 2개이기 떄문이다.
    : 튜플에서 x,y 나누어 받지 않고도 받을 수 있는데  이왕이면 x,y 애초에 분리해서 받는게 좋으니까 분리해서 받는다
    : 코드의 이해를 돕기위해 나누어 받지 않는다고 하면) train, valid = fashion_mnist.load_data()
        : train == (x값들을 요소로 하는 numpy.ndarray, y값들을 요소로 하는 numpy.ndarray) 로 구성된 튜플
            : train[0] == x값들을 요소로 하는 numpy.ndarray
                :train[0][0] == x값들 중 첫번째 값
            :train[1] == y값들을 요소로 하는 numpy.ndarray
                :train[1][0] == y값들 중 첫번째 y값
                :x_train , y_train 으로 애초에 분류해받았을때 y_train[0]으로 y값 라벨 하나 볼 수 있었던것은 
                y_train 이란 변수 == y값들을 요소로 하는 numpy.ndarray 였기 때문이다
'''

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train,y_train), (x_valid,y_valid) = fashion_mnist.load_data()
x_train = x_trian /255
x_valid = x_valid /255
x_train.min(), x_train.max()

#3 모델링

model = Seqential([
    Flatten(input_shape(28,28)), #입력층
    Dense(1024,activation = 'relu'),
    Dense(512,activation = 'relu'),
    Dense(256,activation = 'relu'),
    Dense(128,activation = 'relu'),
    Dense(64,activation = 'relu'),
    Dense(10,activation = 'softmax'), #출력층
])



#4: 컴파일
print(y_train[0])
model.compile(optimizer= 'adam' , loss = 'sparse_categorical_crossentropy',metrics = 'acc')


#5:fit
checkpoint_path='haha.ckpt'
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights = True,
                             save_best_only=True,
                             monitor = 'val_loss',
                             verbose=1)
history = model.fit(x_train,y_train,
                    validation_data= (x_valid, y_valid),
                    epochs=20,
                    callbacks = [checkpoint],)

model.load_weights(checkpoint_path)

#6
'''
만약 evaluate를 통하여 검증하고 싶다면 m=fit의 validation_data의 인수를 쓰는거다
'''
model.evaluate(x_valid, y_valid)


