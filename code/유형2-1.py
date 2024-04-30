#유형 2-1: 지도학습을 이용한 분류기 만들기 - 정형데이터 사용
'''
아이리스 사용
'''

#1 :import
'''
이미지(2차원)를 연산하는게 아니기 때문에 Flatten 은 필요 없다
'''
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint



#2: 전처리
'''
split 이 따로 valid 와 train 으로 구분되어있지 않아 직접 train 을 나눠서 train, valid 용으로 
직접 split 을 만들어야된다.

문제서 주어지는 전처리 코드(data = tfds.load("iris", split=tfds.Split.TRAIN.subsplit(tfds.percent[:80])))는 
구식이므로 다음과 같이 쓴다: 
train_dataset = tfds.load('iris', split='train[:80%]')
valid_dataset = tfds.load('iris', split='train[80%:]')
    :왜 그런지 모르겠지만 %를 사용하여 슬라이싱한다

문제의 어느부분에서 원핫인코딩해야된다고 언급했는지 모르겠지만 어쨌뜬 원핫인코딩 하라고 한다



이렇게 분리한 정형데이터세튼 이터레이터이다
    : 상위데이터세트 단위로 접근해서 뭘 할 생각을 하면 안되고 for문과 take 를 활용해 하위데이터단위로 접근해야된다
        :아예함수를 만든다. (process 같은)
            :하위데이터마다 적용시킬건데 일일이 적을 순 없으니까
            : 쓰임
                : x,y 데이터 분류 
                : 원핫인코딩
                
            : 반환
                : return x,y 형태로 x,y값을 묶어 반환한다
                :이미지 데이터 와 다르게 명시적으로 x_valid , y_valid 이런식으로
                명시되어 데이터가 분류되는게 아니고 valid 하나의 데이터 내부에서 값이 x,y 로 분류된다


'''
train_dataset = tfds.load('iris', split='train[:80%]') #80%
valid_dataset = tfds.load('iris', split='train[80%:]')  #20%


#원핫인코딩해얃되나 알아보기 위하여 몇개 출력
for data in train_dataset.take(5):
    x= data['features']
    y= data['label']
    # 출력내용: x or y 데이터, shape(x데이터 features 개수, ) , 데이터 타입
        # 그러니까 shape 의 첫 인수로 나오는 숫자가 sequential 의 첫 레이어의 input_shape()의 첫 인수로 들어간는 얘 인거임
        # y데이터 부분이 [0,1,0,0,0] 이런식으로 안되있다? (그냥 정수다?) 그럼 원핫인코딩 안되있는거
    print(x)
    print(y) #원핫인코딩 안되있는걸 확인 가능

#추후에 map 으로 상위데이터세트와 엮어서 사용시킬꺼임
def process(data):
    x= data['features']
    y= data['label']
    y= tf.one_hot(y , 3)
    return x,y

batch_size = 10
valid_data = valid_dataset.map(process).batch(batch_size)
train_data = train_dataset.map(process).batch(batch_size)

#3 모델 정의
'''
이미지가 아니라서 Flatten layer 는 필요 없다
x의 feature 수가 4개이므로 4로 한다.
'''
model = Sequential([
    Dense(512, activation='relu', input_shape=(4,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax'),
])

#4 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#5 학습
checkpoint_path = "my_checkpoint.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)

history = model.fit(train_data,
                    validation_data=(valid_data),
                    epochs=20,
                    callbacks=[checkpoint],
                   )

model.load_weights(checkpoint_path)