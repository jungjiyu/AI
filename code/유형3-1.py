'''
horse or human - IDG 사용

특:
rps 처럼 하나의 url 로 모든 원본 데이터셋을 받는게 아니라
2개의 url 로 validation용 train 용 따로따로 받아서 IDG 의 스펙에 validation_split 을 생략한다. 따라서 flow_from_directrory 에서도 subset 생략한다

만들어야하는 IDG 의 종류도 2개이다. 보이진 않지만 문제에서 요구한다고 하는 것 같다
    1. training_IDG : 일반적인 IDG
    2. validation_IDG : rescale = 1/255 스펙만 있는 초간단 IDG
        : 일단 이렇게 만들라고는 해서 만드는데 추측하건데 말그대로 검증만을 위한 거니까 전처리 과정에서
        별도로 증강시키진 않는 것 같다

'''

#1: import
import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

#2: 전처리
'''
다운로드한 폴더의 종류가 두가지 이기에 extractall 부분에서 그냥 'tmp/' 하면 안되고
'tmp/horse-or-human/' , 'tmp/validation-horse-or-human/' 으로 하위 폴더 하나씩 더 만들어주고 각각 그 경로를 대입해야된다.
그냥 'tmp/' 만 쓰면 두 파일이 tmp 아래에서 그냥 짬뽕된다.
'''
_TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
_TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"

urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')

local_zip = 'horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/horse-or-human/')
zip_ref.close()

urllib.request.urlretrieve(_TEST_URL, 'validation-horse-or-human.zip')
local_zip = 'validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/validation-horse-or-human/')
zip_ref.close()


TRAINING_DIR ="tmp/horse-or-human/"
VALIDATION_DIR ="tmp/validation-horse-or-human/"

#정의해야하는 IDG 종류가 2개
training_IDG = ImageDataGenerator(
    rescale = 1/255,
    rotation_range = 10,
    width_shift_range=0.5,
    height_shift_range=0.3,
    shear_range=0.4,
    zoom_range=0.5,
    horizontal_flip=True,
    fill_mode='nearest',
    # 별도로 validation_split이 필요 없다
)

validation_IDG = ImageDataGenerator(rescale = 1/255)

training_dataset = training_IDG.flow_from_directory(TRAINING_DIR,
                                                 batch_size = 32,
                                                 target_size=(300,300),
                                                 class_mode='binary',
                                                #validation_split 안했으므로 subset 도 없다
                                                 )

validation_dataset = validation_IDG.flow_from_directory(VALIDATION_DIR,
                                                        batch_size=32,
                                                        target_size = (300,300),
                                                        class_mode='binary',
                                                        #validation_split 안했으므로 subset 도 없다


#3. 모델 정의
'''
model.summary() 로 마지막 임미지 사이즈 보면서 cnn 쌓기
'''
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),

    Dropout(0.5),
    Dense(512, activation='relu'),

    Dense(1, activation='sigmoid'),

])


#4. 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#5. 학습
checkpoint_path ="my_checkpoint.ckpt"
checkpoint =ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)
history = model.fit(training_dataset,
                    validation_data=(validation_dataset),
                    epochs=20, #솔직히 epoch 30 으로 하면 0.21993 까지 나오는데 최대한 val_loss 비슷하게 하기 위해 조금 적게 설정
                    callbacks=[checkpoint],
                    )
model.load_weights(checkpoint_path)