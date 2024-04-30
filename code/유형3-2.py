"""
cats and dogs
"""
'''
아무리 초기라도 val_loss 가 50% 를 넘어간다 --> 거의 찍는 수준이다
 
IDG 가 아닌 tfds 사용하고 , 꽤 처리가 복잡하다. 따라서 전이학습을 사용하자
'''
#import
'''
다른 부분은 iris 와 똑같고 
모델 정의 하는 부분만 바뀌는거다. 전이학습이용한걸로.
# 전이학습모델 import <--- tensorflow 의 applications 모듈에 다양한 전이학습모델있다 (추천:regnet, inseption )
'''
import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16