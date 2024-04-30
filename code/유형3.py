'''
:컬러사진의 처리
:CNN 사용

유형3
    type a : IDG 사용 (image data generator)
        : 예제 3개
            : rps,horse_or_human
    type b : tfds 사용 (iris 같은거)
        : 예제 3개
            :cats_and_dogs

'''


# import
"""
urllib.request : url 통하여 파일을 다운로드하게 해줌
zipfile: 압축파일관련수행
IDG 는 이미지 전처리 관련이므로 preprocessing -> image 아래에 있다
"""
import urllib.request
import zipfile
from tensorflow.keras.layers import Conv2D, MaxPooling2D ,Flatten, Dropout, Dense
from tensorflow.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

#2 전처리
"""
1.데이터셋다운로드) 자동으로 문제에서 주어짐
    : url 을 변수에 저장하여 사용
    : url로 파일 다운
        : 이때 설정한 파일명이 각 라벨폴더의 상위폴더가됨
    : 압축파일 풀기
    : 압축된 파일 푼것을 특정상위폴더에 저장
        :flow_from_directory 에서 입력해야할 경로는 "특정상위폴더/설정한 파일명/" 이 됨

2. IDG 정의
    : rps 이미지들은 variation 이 그리 크지 않으므로 범위를 좁게 설정
3. flow_from_directory
    : 정의한 IDG 와 앞서 저장한 데이터셋의 경로를 사용하여 val, train 용 데이터 만듬
"""

url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
urllib.request.urlretrieve(url, 'rps.zip') #rps 라는 상위폴더에 해당 데이터를 저장
local_zip = 'rps.zip' # 압축된 파일을 변수에 저장
zip_ref = zipfile.ZipFile(local_zip , 'r') # 압축을 품
zip_ref.extractall('tmp/') # rps.zip 압축 푼것을 tmp 에 저장
zip_ref.close()

TRAINING_DIR = "tmp/rps/" #flow_from_directory 에 쓰일 상위폴더의 경로

IDG = ImageDataGenerator(
    rescale = 1/255, #이미지 정규화
    rotation_range  = 5, # 0~5도 회전
    width_shift_range = 0.05, # 좌우폭
    height_shift_range = 0.05, # 위아래폭
    shear_range = 0.02, # 굴절
    zoom_range = 0.02, # 확대
    horizental_flip = True, # 횡방향반전
    fill_mode = 'nearest', # 주변이미지로 채움
    validation_split=0.2 # train = 80% , val = 20%
)

training_dataset = IDG.flow_from_directory(TRAINING_DIR,
                                        batch_size = 32, #문제에서 주어짐
                                        target_size = (150,150),  #문제에서 주어짐
                                        class_mode= 'categorical', # softmax 이면 무조건
                                        subset = 'training',
)

validation_dataset = IDG.flow_from_directory(TRAINING_DIR,
                                          batch_size = 32,
                                          target_size = (150,150),
                                          class_mode='categorical',
                                          subset= 'validation',
)

#3 모델정의: 층쌓는것
'''
cnn 쓰는 경우 model.summary() 를 했을때 cnn 관련 마지막 레이어에서 이미지의 사이즈는 5*5 ~15*15 여야 적당
Dense 에서와 마찬가지로 Conv2D에서도 activation 을 쓰고, 이를 합쳐서 한꺼번에 표기한다
    : 그런데 연산도 안일어난다면서 왜 필요한지는 모르겠다
한변의 사이즈가 n인 maxpooling 적용후 : 가로세로 1/n배씩
필터사이즈가 3*3 인 conv2d 적용후 :가로세로 -2씩
'''

model = Sequential([
    Conv2D(64,(3,3),activation='relu',input_shape = (150,150,3),# 필터개수,필터사이즈,활성함수
    MaxPooling2D(2,2), # 2*2 의 사이즈로 2칸 씩 건너뛰면서
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128,(3,3),activation = 'relu'),
    MaxPooling2D(2,2),
    Conv2D(128,(3,3),activation = 'relu'),
    MaxPooling2D(2,2), #마지막 CNN계열 레이어
    Flatten(), # 이미지를 평탄화
    Dropout(0.5), # 50%의 확률로 학습시켜 과적합을 방지
    Dense(512,activation='relu'),
    Dense(3, activation = 'softmax'),
           )
])

print(model.summary()) # CNN 적정량 쌓은건지 확인

#4 :컴파일 (도면으로 집짓기)
'''
IDG를 사용하여 생성된 데이터를 대상의 경우  원핫인코딩 자동으로 되기때문에
softmax 썼으면 백퍼 categorical_crossentropy 임
'''
model.compile(optimizer  = 'adam' , loss= 'categorical_crossentropy', metrics=['acc'] )

# 학습(fit)
'''
순서 기억

1. 체크포인트를 저장할 체크포인트 경로를 변수에 저장해둠
    : 단순히 이런다고 경로가 생성되는건 아니고 ModelCheckpoint 의 인수로 들어가야 거기에 진짜 생성되는거임
    근데 그 이후에도 꽤 쓰일거니까 변수에 저장해두는것
2. 저장해놓은 체크포인트 경로에 체크포인트 모델생성
3. fit의 인수로 체크포인트 넣어 학습시킴
4. 학습 완료 이후 체크포인트 반영시킴
'''
checkpoint_path = "mycheckpoint.ckpt"
checkpoint = ModelCheckpoint(filepath ="mycheckpoint.ckpt",
                             save_weights_only = True,
                             save_best_only=True,
                             monitor = 'val_loss',
                             verbose = 1)
history = model.fit(training_dataset,
                    validation_data = (validation_dataset),
                    epochs = 20,
                    callbacks=[checkpoint],
                    )

model.load_weights(checkpoint_path)