#1차시

##2차시 부터는 다시 잘 봐라
"""
y= wx+b
    : w == 가중치 == 기울기 == weight
    : b == 편향 == bias

가설함수: x,y 데이터 세트 있을때 해를 구할 수 있는 잠정적 수식.
    :w 와 b를 구하는 것


loss == 손실 == cost (== error ==오차)
    : loss 는 대부분 mse (mean squared error,평균제곱오차)로 구한다
        : ' (실제값-예측값)**2 '평균의 sum
        : 그냥 '실제값 - 예측값'의 sum 이 아닌것은 음수 , 양수로 상쇄되어 별로인 상태를 좋은 상태라 칭할 수 있기 때문

    : 처음에는 급격하게 줄어든다
        : 기울기의 변화가 처음에는 급격해서 값이 급격히 변하는데 나중에 정답의 근처로 갈수록 완만해지니까
         (당장 경사하강법의 수식을 생각해보라:w = w- a*w')

loss (mse)를 0 에 가깝게 만드는게 목표

경사하강법
    : 딥러닝에서 가장 중요한 최적화 알고리즘
        : 최적화 알고리즘 == 오차를 0 에 가깝게 만드는 w,b를 구하는 알고리즘

    : 오차함수가 주어졌을때 경사를 타고 내려와 최저점에 도달하는 최적화 알고리즘
    : 지점을 한방향으로 옮겨가며 (기울기를 다르게 하며) 값을 업데이트
        : w = w- a*w' 의 식을 사용하여 업데이트
            : w' =  gradient = 기울기 = 미분값
            : a = learning_rate == 학습률 : 보폭
                : 얼마나 큰 보폭으로 지점 찍어 기울기 구할건지 설정하는 것
                : Decent learning rate(적당한 학습률) 로 설정해 주어야
                :  학습율이 너무 작으면 넘 오래걸리고 너무 크면 minimum 도달 불가

keras : 텐서플로우 쉽게 쓸 수 있게 도와주는 고수준 API
    :tensorflow 의 하위 라이브러리
    :from tensorflow.keras ..
    :참고로 대문자 아니다


하이퍼파라미터 : 우리가 넣을 수, 설정할 수 있는 값
    :epoc이나 학습률 같은거
    :acc 같은 결과? 산출값?은 하이퍼파라미터라고 하지 않음


acc == accuracy == 정확도
    : 에폭 증가함에 따라 증가

지도학습 == supervised learning
    : 입력,출력데이터 set 를 주고 학습시키는 방법
        : x 데이터 == features 데이터 (== input 데이터)
        : y 데이터 == label 데이터 == target 데이터 (==output 데이터)

    : 만들 수 있는 모델
        1. 분류 모델
            : 분류대상)
                (1) 이미지데이터
                (2) 정형데이터 : 엑셀처럼 행과 열로된 데이터
        2. 회귀 모델 : 특정 값을 예측

batch size: 학습 단위. 한번에 학습하는 양. 업데이트 단위
    : 우리가 지정할 수 있는 하이퍼파라미터
batch : batcH size를 적용했을때 해야하는 학습횟수 . 업데이트횟수
    : 필요한게 용량이 큰 데이터인 경우 gpu 에 한번에 못집어넣음
    : epoc 이 한번 끝날때마다 업데이트되는게 아니라 batch 한번 끝날때마다 업데이트 되는거다
    : 모델의 성능
        ex) 1000장
            : batch size = 100 장 , batch = 10 개
                : 한번에 학습되는(업데이트되는) 양이 많다.
                    : 빠르게 업데이트 되지만(미분을 계산해야되는데 10번만 계산하면 되니까)  왔다리 갔다리 큰 폭으로 진행되어(급격히 값이 변하니) 정교함이 떨어진다(val_loss 크다)
            : batch size = 10 장 , batch = 100 개
                : 한번에 학습되는 양이 적다
                    : 느리게 업데이트 되지만 값이 정교하다(val_loss 가 적다)
    : 논문에 따르면 일단 batch 사이즈는 최대한 작게 하는게 좋다. 시간이 오래걸리는게 문제이지.
    :1004장 인데 batch size 가 10 이되어 나누어떨어지지 않는경우
        : 몫 + 1 이 batch 가 된다. 즉, 나머지까지 처리될때까지 돌려야된다

참고: (매개변수와 같은걸) 소수점으로 받아야된다 ? 퍼센테이지로 적용될 확률 크다

Dense 레이어: 골조
CNN : 이미지 분리기에 쓰이는 레이어 ---> 화려
    : 컴퓨터비전분야에 잘 쓰임



딥러닝 학습 순서
    1. 쓸 모듈 import
        tensorflow.Keras.models
            :Seqential()

        tensorflow.Keras.layers
            :Conv2D()
            :MaxPooling2D
            :Flatten()
            :Dropout()
            :Dense()
          # 참고로 각각 쓸게 아니라 단순히 Dense 의 인수로 'relu', 'softmax' 이런식으로 설정할꺼면 따로 import 안해도 된다.
            :ReLU()
            :Softmax()
            :Sigmoid()

        tensorflow.Keras.callbacks
            :ModelCheckpoint()

        tensorflow.keras.applications
            : VGG16 와 같은 pretrained 모델이 들어있다
            : 기타 추천 모델) regnet, inseption
        numpy
            :array
                : 리스트를 형변환하기 위해 사용된다.
                : list vs array
                    : list
                        : 파이썬에 기본 존재
                        : 다양한 자료형의 값을 요소로 함
                        : 동적할당임
                        : 이터러블임. 즉 for 문에 넣을 수 있음

                    : array
                        : 배열임. 파이썬에 기본 존재가 아니라 numpy 모듈을 불러와야 사용 가능
                        : 요소의 자료형은 하나로 통일해야됨
                        : 정적할당임 . 즉 사이즈를 변경할 수 없음
                        : 이터러블이 아님. 즉 반복대상이 아님
                        : numpy 가 수학관련 모듈인 만큼 연산에 유리함

                    : 리스트를 배열로 형변환 ) np.array(리스트)
                    : 배열을 리스트로 형변환 ) 배열.tolist()

        tensorflow.keras.datasets
            : 이미지 데이터세트
                :fashion_mnist
                :mnist
            : 정형데이터세트
                :iris

        tensorflow.keras.preprocessing.image
            : ImageDataGenerator()

        urllib.request
            :urlretrieve(url, '파일명.zip')
                : url 로 파일을 다운로드 받을 수 있게 함

        zipfile
            : 압축된파일을푼것변수 = ZipFile('파일명.zip', '모드')
                : 압축된 파일을 푼것을 리턴
                : 모든는 'r' 'w' 같은거
                : extractall(경로)
                    : 압축된파일을푼것변수.extractall('경로') 의 꼴로 사용하여 압축된 파일 푼 것을 어느 상위파일에 저장할지 결정
                : close()
                    :압축된파일을푼것변수.close() 로 해당 폴더에 관한 행위 종료

    2. 데이터 전처리 : 사진 사이즈 조정같은
        : 내가 만든 데이터 명칭
                하위데이터: 하위데이터세트의 x 혹은 y값 하나
                    ex) y_valid[0]

                상위데이터: x 혹은 y값의 덩어리
                    ex) y_valid

                하위데이터세트: 데이터를 짝지은 것. x데이터 하나와 y데이터 하나를 묶은것
                            : 그러니까 상위데이터세트의 요소
                    ex) for data in train_dataset.take(1) 에서 data

                상위데이터세트 : 하위데이터세트를 요소로 하는 덩어리
                    ex) train_dataset = tfds.load('iris',split ='train[:80%]' )

            : 데이터 세트의 개념 중 split 이란게 있다
                : split == 데이터가 어떤 카테고리의 데이터세트로 구성되어있는가
                    : ex) 'train' , 'valid'

        : 데이터종류
            (1) tensorflow.keras.datasets 를 사용하는 데이터 세트
                1. 이미지 데이터세트
                    :이미지데이터세트의 경우 딱히 따로 import 안하고 걍 전체를 쓴다
                    :split 이 보통 'train' 카테고리 뿐 아니라 'valid' 카테고리까지 있다
                        : 하나의 변수로 받지 않고 두개의 변수로 나누어받는다.
                            : 이미지 데이터세트의 경우 두개의 단순 변수만으로 받기보다는 x,y 를 요소로하는 2개의 튜플로 받는게 유리하다
                                : 왜 이게 가능할까) 튜플을 반환하니까 튜플로 받을 수 있다
                                    : 일단 train, valid = fashion_mnist.load_data() 하면
                                    #train == (x값들을 요소로 하는 numpy.ndarray, y값들을 요소로 하는 numpy.ndarray) 로 구성된 튜플
                                    print(type(train))
                                        #train[0] == x값들을 요소로 하는 numpy.ndarray
                                        print(type(train[0]))
                                            #train[0][0] == x값들 중 첫번째 값
                                            print(train[0][0])
                                        #train[1] == y값들을 요소로 하는 numpy.ndarray
                                        print(type(train[1]))
                                            #train[1][0] == y값들 중 첫번째 y값
                                            print(train[1][0])
                                                :x_train , y_train 으로 애초에 분류해받았을때 y_train[0]으로 y값 라벨 하나 볼 수 있었던것은
                                                y_train 이란 변수 == y값들을 요소로 하는 numpy.ndarray 였기 때문이다

                            :기억할 것은 x_train , y_trian은 배열(array)이라는 것이다.
                                :배열이므로 이터레이터가 아니라 for문의 반복대상이 될 수 없다.
                                :인덱스를 통하여 각각의 요소에 접근한다
                                    ex) y_train[0] ---> y_train 이라는 배열에서 0번째 요소에 접근

                2. 정형데이터세트(==structured)
                    :정형 데이터 세트의 경우 import 해준다
                    : split이 보통 'train' 카테고리만 있어 데이터 받을때 하나의 변수만을 사용하여 받아야되고 슬라이싱을 통해 학습용과 검증용으로 직접 나눠줘야된다.
                        : train[80%:] 와 같이 그냥 [0.8:]이 아니라 백분율식으로 분할해야한다.(안그럼오류남)
                        : 이미지데이터처럼 애초에 x,y값 분리해서 받을 수 없는 이유((train_x,train_y) = tfds.load('iris', split='train[:80%]') 같은게 안되는 이유)
                            :튜플이 아닌 이터레이터를 할당받으니까
                                : 탐구)
                                    train_dataset = tfds.load('iris', split='train[:80%]')
                                    print(type(train_dataset))#train_dataset==정체불명의 이터레이터

                        : for과 take 를 통해 키를 활용해서 x,y값에 접근)딕셔너리를 반복자로 하는 이터레이터여서 각각의 딕셔너리에 접근후 키값을 이용해 접근해야된다.
                            :탐구)
                                for data in train_dataset.take(1):
                                print(type(data))#train_dataset 은 딕셔너리를 반복자로 하는 이터레이터다

        :데이터 전처리 종류
            1. (tensorflow.keras.datasets 를 통해 )불러올 수 있는 데이터 세트
                (1) 데이터세트 불러오기
                    1. 이미지데이터세트:   데이터세트명 = tensorflow.keras.datasets.데이터세트명
                        ex) fashion_mnist = tensorflow.keras.datasets.fashion_mnist
                    2. 정형데이터세트: 추가적으로 없음

                (2) 학습용, 검증용 데이터 분리 && x,y 데이터 분리
                1. 이미지데이터세트
                    : load_data() 이용하고, 애초에 x,y 로 나누어 받음
                        : 애초에 반환값이 x배열 ,y 배열 을 요소로 하는 튜플이라서 그럼
                    :ex) (x_train,y_train),(x_valid,y_valid)= fashion_mnist.load_data()
\
                2. 정형데이터세트 분리하고 변수에 할당
                    : load('데이터세트명' , split = '카테고리명[백분율% 슬라이싱]' ) 이용하여 x,y 퉁처서 받음
                        : 반환값이 튜플 리스트 그런게 아니라 딕셔너리라서 자동매치가 안되고
                        추후 반복문을 통하여 하위데이터 단위로 접근하고 내부적으로 x,y 분리해줘야됨.

                    :ex)
                    train_dataset = tfds.load('iris' , split = 'train[80%:]') #물론 import tensorflow.keras.datasets as tfds 한 이후에 tfds 쓴거다
                    valid_dataset = tfds.load('iris' , split = 'train[:80%]') # 80% 의 나머지 즉 , 끝 20%


                        : shape 를 사용 ---> 상위데이터의 개요((이미지개수,세로픽셀,가로픽셀)) 반환

                    2. 정형데이터세트의 x 혹은 y 데이터에 찐 접근
                        : 반목문을 사용하여 접근
                        : (반복문을 통하여 하위데이터 단위로) 데이터 자체를 출력
                            : ( x혹은 y 데이터 ,shape=(,,), 데이터타입)
                                : 솔직히 데이터타입은 별로 신경안써도 됨
                                : 이미지데이터에서 .shpae 를 써야 나왔던게 그냥 x,y데이터만 출력해도 한꺼번에 나오므로 굳이 정형데이터에선 x.shape 같은거 따로 안한다
                            :ex) for data in train_dataset.take(1):
                                    x= data['features']
                                    y = data['label']
                                    print(x) --> tf.Tensor([5.1 3.4 1.5 0.2], shape=(4,), dtype=float32) : 후에 input_shape(4,) 로하면 된다
                                    print(y) ----> tf.Tensor(3, shape=(), dtype=float32) : 달랑 3 인걸 봐선 원핫인코딩 안된 상태이다 . x값도 아니므로 (따로 input_shape랑 엮인게 없으므로 당연히 shape 부분은 비어있다)
                         : how)
                        하위데이터세트를 매개변수로 하여 x,y 값을 분리하는 함수를 따로 정의하고
                        map 함수를 이용해 상위데이터세트에 이 함수를 적용시켜 하위데이터세트단위로 x,y 값을 분리한 이터러블객체를 get 한다
                            : ex)
                            train_dts = tfds.load('iris' , split='train[:80%]')
                            def process(dts):
                                x = dts['feature']
                                y = dts['label']
                                return x,y
                            classified_dts = train_dts.map(process).batch(10)
                            #classified_dts 는 내부적으로 x,y값이 분리된 이터러블

                        : 뻘짓주의
                            뻘짓 1 ) load 될때 하나로 퉁쳐서 받고 추후에 x,y 분류해야된다
                                :train_x , train_y = tfds.load('iris', split = 'train[:80%]') 오류난다
                            뻘짓 2 ) 상위데이터세트에서 바로 키값을 뽑아내서 x,y 분리하려고하면 안된다
                                :train_x =train_dataset['features']
                                train_y = train_dataset['label'] 오류난다
                                    : 왜 오류날까? ) 생각을 해라 생각을
                                    :
                                        각각의 데이터 셋은 'features' 와 'label' 을 key값으로 가지는 딕셔너리이다
                                        정형데이터는 딕셔너리를 요소로하는 이터러블이다
                                        물론 이와 완전히 같진 않겠지만 대충 흉내내보면
                                            dict_1 = {'apple':1,'bannana':2}
                                            dict_2 = {'apple':3,'bannana':3}
                                            dict_3 = {'apple':4,'bannana':4}

                                            train_dataset = [dict_1, dict_2, dict_3]
                                            t_x= train_dataset['apple']
                                            t_y= train_dataset['bannana']
                                        이게 되겠냐? train_dataset가 딕셔너리도 아닌데?
                                        파이썬한테 뭘바라는거냐 너는


                (3) 데이터 원핫인코딩: tf.one_hot( 원핫인코딩대상 , 라벨개수)
                    : 물론 x데이터는 원핫인코딩안한다. 라벨값인 y데이터만 원핫인코딩한다
                    1. 이미지데이터세트 원핫인코딩
                        : 그냥 상위데이터 대상으로 적용시키면 됨
                            ex)
                            onehot_train = tf.one_hot(train_y , 10)
                            onehot_valid = tf.one_hot(valid_y , 10)

                    2. 정형데이터세트 원핫인코딩
                        : 반복문을 사용해 하위데이터세트 단위로 함.
                            : 당연한게 x, y 구분한 결과가 이미지데이터세트처럼 x_vaild , y_train 이런식으로
                            나왔던게 아니라 내부적으로 구분된거라서(그런니까 당장 눈에보이는 변수단위로 분리된게 아니라서)
                            y데이터를 원핫인코딩 하기 위해서는 내부적으로 접근해줘야된다.

                (4) 이미지 정규화: 255 로 나눔
                        : 픽셀이 0~255 의 값을 가지는데 이를 0~1 로 만들어 주면 더 빠르고 정교한 처릴 할 수 있어서 그렇다
                        : 물론 이미지 데이터만을 대상으로한다. 라벨값을 정규화하진 않는다
                        : 왜 그런진 모르겠는데 복합연산자 쓰면 오류나고 x_valild = x_valid / 255 이런식으로 해야된다.

                (5) 이미지 사이즈 조정: 문제에는 안나왔다

            2.image agumentation
                : 이미지 증강
                    : 우리가 임의로 만든 (가상의) 케이스에 대하여 학습시키는거니까
                : 확대 축소 비틀림 반전 같은거 있어도 같은 대상임을 인지시키는게 목적 ---> 기계는 우리보다 연산속도가 빠르니 다양한 케이스를 임의로 만들고 그 케이스를 모두 학습시켜 대비하자
                : imageDataGenerator 모듈 (편하게 IDG라 하자)
                    : 기능
                        1. 이미지 증강: 쉽게 다양한 케이스에 대비하게 해준다
                            : 쉽게 이미지 전처리를 하게 해줌
                        2. 로컬 사진(컴퓨터의 사진)을 코드로 불러옴
                            :flow_from_directory()를 이용한다
                            : jpg , png 같은 형태는 바로 코드에 못넣고 .tensor 와 같은 형태로 저장되어야하는데
                            이걸일일이 코드로 구현할려면 어려운데 IDG는 이를 쉽게 처리해준다
                            :로컬사진 불러오는 원리)
                            1. 라벨 폴더 전체를 아우르는 폴더(상위폴더)의 경로를 입력
                                ex) '아빠', '나' , '엄마' 의 사진데이터가 각각 '아빠', '나' , '엄마' 라는 이름의 하위폴더에 저장되어있고
                                이 폴더들은 '가족'이라는 상위폴더에 저장되있을때 '가족'의 경로를 알려준다
                                    : "C:\사진\가족\"

                            2. 입력하면 하위폴더 단위로 구분해야함이 인지됨 && 각각의 데이터가 tensor 의 형태로 변환됨
                                ex)'아빠', '나' , '엄마' 의 3개의 라벨로 분류해야된다고 인지,
                                tensor 의 형태로 각각의 하위폴더의 이미지를 변환
                    :사용방법
                        1. IDG 정의 --> 어떤 방식으로 증강시킬건지 스펙을 정의해놓는다
                            IDG= ImageDataGenerator(
                            rescale = 1. /255  #정규화
                            rotation_range = 각도,
                            width_shift_range = 소수점값,
                            height_shift_range = 소수점값,
                            shear_range= 소수점값,
                            zoom_range = 소수점값,
                            horizontal_flip = 불값,
                            fill_mode = 옵션,
                            validation_split = 소수점값,
                            )
                            : 괄호 안의 부분부분들(rescale,rotation_range 같은 각각) == 옵션
                                1. rescale : 이미지 정규화
                                    : 몇을 곱할건지 --> 그러니까 보통 x = x/255 하니까 255 로 나누는 것임으로 1/225 를 써줌 됨
                                    : 사진의 사이즈 자체를 조정하는게 아니라 픽셀값 정규화임
                                        : 사진 사이즈 자체 조정은 뒤에 flow_from_directory 의 target_size에서 이루어짐
                                2. rotation_range
                                    : 몇도의 범위로 랜덤하게 돌릴 것인지
                                        : 만약 40 이라면 0~40 도의 범위에서 랜덤하게 사진을 돌린다.
                                3. width_shift_range, height_shift_range
                                    :  0% ~ 인수 % 의 범위로 가로, 세로방향으로 랜덤하게 이동시킨다
                                    : 소수점을 적는다 --> 소수점을 적는다는건  퍼센트 개념으로 움직인다는것
                                        ex) 0.2 --> 20퍼
                                4. shear_range
                                    : 0% ~ 인수 % 의 범위로 랜덤하게 이미지를 굴절시킨다
                                5. zoom_range
                                    : 0% ~ 인수% 의 범위로 랜덤하게 확대
                                6. horizental_flip
                                    : True 면 좌우 반전을 적용시킴
                                7. fill_mode
                                    : 이동같은거 시키고 나서 빈 공간을 무엇으로 채울것인가
                                    : 인수로 쓸 수 있는 여러가지 옵션이 존재
                                        1. 'nearest' : 주변의 픽셀들로 채움
                                        2. 'reflect' : 반전시켜서 채움
                                            ex) ____ABCD --> DCBAABCD
                                8. validation_split
                                    : 검증용 세트의 비율 결정 ---> 학습용은 자동으로 나머지가 됨
                                        ex) validation_split = 0.2 --> 검증용: 20% , 학습용 : 80%

                            : 괄호 안을 퉁쳐서 스펙이라고 한다
                            : 스펙을 결정하는 tip
                                : 데이터셋을 고려하여 결정 . 다양성 적으면 적은 퍼센테이지 범위로, 크면 넓은 퍼센테이지 범위로
                                ex) 증명사진 --> 좌우이동,확대,굴절,확대...의 확률 적음
                                    : 각각 낮은 비율로 설정
                                        ex) 3도 , 0.05 , 0.05 ..
                                ex) 셀카 --> 좌우이동,확대,굴절,확대의 확률 ㄴ높음
                                    : 각각 높은 비율로 설정
                            : 똑같은 방식으로 원본을 증강시킨것을 데이터로 할 것이므로 validation, train 용 각각 만들 필요는 없다

                        2. flow_from_directory ---> 로컬파일의 이미지 가져와 앞서 정의한 IDG의 스펙대로 이미지를 증강시킨것을 저장
                            :training_dataset =IDG.flow_from_directory(큰파일경로 ,
                                                                    batch_size= 배치사이즈,
                                                                    target_size = (가로픽셀,세로픽셀), #이미지의 사이즈를 해당 사이즈화 시킨다
                                                                    class_mode = '어쩌구',
                                                                    subset = 'training',
                                                                    )
                            validation_dataset =IDG.flow_from_directory(큰파일경로 ,
                                                                    batch_size= 배치사이즈,
                                                                    target_size = (가로픽셀,세로픽셀), #리사이즈 시킨다
                                                                    class_mode = '어쩌구',
                                                                    subset = 'validation',
                                                                    )
                            :flow_from_directory 함수 구성)
                                :target_size
                                    : 이미지를 지정한 사이즈로 리사이즈 시킨다
                                        : IDG 의 rescale 이랑 헷갈리면 안된다. rescale 은 정규화다.
                                        그러니까 rescale 에서는 픽셀 사이즈를 줄이고, target_size 에서는 픽셀개수를 뺴든 늘ㄹ리든 해서 개수를 조절하는것
                                    :딥러닝 모델에 들어가는 이미지의 사이즈는 전부 동일해야된다.
                                : class_mode
                                    : 그냥 규칙이다. 외워라
                                    : 모델의 출력층의 activation이
                                        1) sigmoid 였으면 'binary'
                                        2) softmax 였으면 'categorical'
                                            : IDG 쓰면 자동으로 원핫인코딩 처리 다 되기 때문에 sparse 는 고려도 안한다
                                : subset
                                   : training 혹은 validation 을 적는다
                                        : 앞서 IDG 정의했을때 설정한 비율만큼 데이터가 할당된다
                                            만약 IDG 옵션에서 'validation_split' = 0.2 였는데
                                            subset 을 'training' 이라고 했다면 파일경로에 담겨있었던 데이터의 80%가 할당되는것

                                : 참고로 학습용에서는 batch 사이즈 설정이 중요하지만 검증용에서는 미분 업데이트가 일어나지 않아 별로 신경안써도 된다

                        :Found (검증혹은 학습용)전체사진수 images beloning to 라벨수 classses 나왔음 정상적으로 된거다
                            : 라벨수는 물론 앞서 썼던 상위폴더의 하윕폴더 개수를 말하는 것
                            : 정상적으로 처리되지 않으면 전체사진수가 0같이 존나 이상하게 나온다

                        : training_dataset , validation_dataset 이 각각 학습용 데이터 , 검증용데이터가 되는거다
                            : IDG로 생성된 데이터셋이란 이야기
    3. 모델링: 설계
        : model = Seqential([ 레이어, ])
        : Seqential([ ])
            : 차곡차곡 순서대로 쌓는다는 뜻으로 model 의 정의 부분에 쓰임
            : 레이어들을 인수로 한다
                : 모든 레이어들의 끝에 쉼표 붙이는거 잊지 x

        :레이어
            : CNN의 경우 깔때기를 엎어놓은 작은--->큰 의 꼴로 필터의 개수를 설정하고
             Dense 의 경우 깔때기 꽂아놓은 모양처럼 노드의 개수를 큰-->작은 으로 구성한다
            요즘엔 하드웨어가 좋아서 별 상관없을지도 모르겠지만 2의 제곱수로하면 gpu가 처리를 좀더 잘한다고 한다
            :특성추출(CNN) 과 연산(Dense) 중 뭘 먼저해야 유리할까
                : 특성발굴후 연산하는게 유리---> cnn 레이어 먼저 다 쌓고 그 후에 dense 레이어 다 쌓는다
                : 적당한 층의 CNN을 쌓는 tip:
                    : model.summary() 해봤을때 Dense 레이어 넘어가기 직전, 즉 CNN 레이이어계열 마지막 레이어에서
                    이미지의 사이즈가 5*5~15*15 이면 된다.

                : 그래서 cnn 을 포함하는 모델의 구성:
                    Sequential([
                    Conv2D( 필터개수, (사이즈) ,activation = 'relu', input_shape(세로,가로,채널)),
                    MaxPooling2D( 한변사이즈,건너뛰기단위 ),
                    ...,
                    Flatten(),
                    Dropout(소수점),
                    Dense(노드개수,activation='relu'),
                    ...,
                    Dense(라벨개수,activation='어쩌구'),
                    ])

            :전이학습: transfer learning
                : 방대한 데이터로 사전학습된 모델(pretrained model)의 가중치를 가져와서 가지고 있는 데이터 셋에 맞게 약간의 튜닝을 주면 된다.
                    : 참고로 facebook은 가중치를 공개하고 있는데 chatgpt는 안그러고 있다
                : 모델 정의 부분만 손봐주면 된다. (물론 해당 사전학습모델을 사용하기 위한 모듈도 추가적으로 import 해주긴 해야됨)
                 전이학습을 적용한다는 것은 크게 어려운게 아니다.
                :유명한 모델의 공개된 구조도대로 설계해서 cats and dogs 데이터에 사용해보면 그 성능이 안나옴
                    :왜? ) 데이터의 수량과 퀄리티가 달라서. vgg net 은 굉장히 방대한 데이터로 복잡한 연산을 수행하는데 우리는 그러한 데이터를 가지고 있지 않다.
                        : 일반인 수준에선 그러한 퀄리티의 방대한 데이터 확보가 어렵다
                    :그럼 구조도를 알아도 전혀 쓸데 없는것?
                        : 아니다. 전이학습이란 개념이 있어 개인도 딥러닝 모델 만들 수 있다
                : 사전학습된모델은 특정 데이터셋에만 사용할 수 있는게 아니다.
                    :기존 학습데이터셋과 관련있는 데이터셋에 적용시키면 성능이 굉장히 좋게 나오긴 하겠지만,
                    관련없는 데이터셋으로 학습을 진행시킨다 하더라도 특성을 잘 발굴해내는 기능은 그대로이기 때문에 많은 사람이 전이학습을 택하는 것이다.

                : 사용방법
                    1. 해당 모델 import
                        : from tensorflow.keras.applications import 어쩌구모델명
                    2. 사전학습모델 세팅
                        (1) 기본적인 설정으로 사전학습모델 생성
                            : transfer_model = 어쩌구모델명(weights = '저쩌구', include_top = '불값' , input_shape=(문제맞춰서 잘))
                                :weights ) '저쩌구' 데이터셋으로 학습된 가중치를 가져와라
                                    : None 이면 구조만 가져오고 가중치는 안가져오는것
                                : include_top) CNN 처리 레이어 이후 부분도 가져올 것인지
                                    : 'False' 이면  CNN 처리 부분만 가져오고 뒤의 Dense 부분은 따로 안가져온다
                                    : 솔직히 모델들마다 어디까지를 top이라고 하는건지 모르겠다
                                : input_shape) 모델 정의 부분의 첫 레이어로 들어가게 됨으로 input_shape를 이때 지정해주어야된다.
                        (2) 생성한 사전학습모델의 가중치 freeze 여부 설정
                            : transfer_model.trainable = 불값
                                : 'False' 이면 받은 모델의 가중치를 freeze 시킨다는것. 즉 우리가 만든 모델의 학습이 진행되어도 그에따라 가중치가 변경되진 않는다
                                    : 해야되는 이유가 질 낮은 데이터로 물흐릴 필욘 없으니까
                                    : 우리는 이렇게 설정하고 뒷부분, 즉 dense 부분의 가중치만 학습 시킴에 따라 업데이트 시킬꺼다
            :종류
            1.Dense(n , activation = '어쩌구') : Fully Connected Layer (==Dense Layer)
                : n 은 해당 레이어 노드의 개수
                : 가장 근본적인 레이어
                : 신경==neuron==node : 가중치(w)를 가지고 있다
                : 뉴런들끼리 모두 연결되있기 때문에 fully connected 이다.
                : 단방향으로 이동한다.
                : 연산 작용하는 역할하는 레이어
                    : 이름에서 엿볼 수 있듯이 fully connected layer 이다보니 모든 픽셀 대 픽셀 끼리 모든 연산을 한다.
                    즉 대상과 전혀 상관없는 픽셀들끼리까지 연산하여 불필요한 연산이 많고 성능은 오히려 떨어진다.

                : 구성
                input layer:
                    input_shape[한번에 들어가는 x데이터의 개수] :데이터의 입력형태를 알 수 없어 명시적으로 신호가 몇개가 오는지 알려줘야됨
                        : 그러니까 [] 안에는 매개변수 개수가 들어가는 것
                        : ex)  이미지 --> 이미지의 픽셀 개수들어감
                        : ex)  정형데이터 --> 데이터의 feature 개수 들어감
                        ex) 9 --> 10 으로 y값 하나당 x값 힌개 매치되면 [] 안에는 1임

                hidden layer:
                    임의로 조정가능한 레이어. 의도따라 복잡하게 , 간단하게 만들 수 있음
                        : 무조건 복잡하다고 좋은게 아님. 간단한 연산 수행 목ㅈ적이면 간단하게 만들어야 오히려 성능 좋음
                output layer:
                    :n , 즉 아웃풋 레이어 노드의 개수는 라벨의 개수(분류대상의 개수)와 같아야 됨.

            2. activation(=활성함수=비선형함수)
                :없어도 되긴하는데 모델의 복잡도를 높이고 싶으면 중간중간 써주어야한다
                    : Dense는 선형함수로, Dense 만을 썼을때는 그 복잡도가 높지 않지만
                    activation, 즉 활성함수(비선형함수)를 각각의 Dense 마다 (뒷부분에) 끼워넣게되면은 그 복잡도가 증가하여
                    결론적으로는 복잡한 연산을 수행할 수 있는 모델을 만들 수 있다
                        : 일일이 한줄로 다 안쓰고 Dense쓸때 아예한묶음으로 쓸 수 있다
                            : 일일이 쓴다 하면 선형함수 다음에 써야됨
                :종류
                    1. ReLU()
                        : 히든레이어의 activaion으로 쓰임
                        : Dense 층에 간략화하여 들어갈때는 'relu' 로 들어감
                        : 함수형태) 0이하에서는 y=0 , 0 이후로는 y=x

                        :오차역전파(backpropagation)
                            : 사실 tensorflow 내부에서 알아서 다 해주기 때문에 굳이 다 알필욘 없지만 알고 있음 좋다.
                            : 반댓말은 순전파.
                            : forwardpass) 부모에대한미분값구할때방향, backwardpass) chainrule 이용 거슬러올라오는방향
                            : 아웃풋 레이어에서 각각의 노드에서 내뱉어진 weight 들이 정답값과의 오차를 0에 가깝게 만드는것
                                : 이렇게 만들기 위해선 그 이전 노드들의 가중치를 미세조정하여 최적의 세팅을 해놔야됨
                                : 뉴런들의 가중치를 업데이트 하는법
                                    : 경사하강법도 나쁘진 않지만 층이 복잡한 경우 모든 미분값을 구하긴 복잡
                                    : 오차역전파를 이용하면 복잡한 미분 없이도 복잡한 층에 대하여 가중치 업데이트 가능
                            : 오늘날의 딥러닝 모델을 만들 수 있게 해줌
                            : 원리
                                : 미분 값을 구할때는 저멀리의 미분값을 한방에 구할 수 없음
                                    : 당장 생각해봐도 이계도 함수라는 개념이 존재. 두번 미분한 값을 구하기 위해서는 미분을 2번 해야됨/
                                    : 직접적으로는 자기 부모에대한 미분값만 구할 수 있음
                                : 자기 부모에 대해 편미분한 값을 각각 저장해둠. 최종보스부모노드(?) 직전까지. 그후 chain rule 이용
                                    :chain Rule:  (끝쪽의)특정노드1에 대한 (앞쪽의)특정노드2의 미분값 == 특정노드1 바로이전 노드부터 특정노드 2까지 타고 내려가는 노드에 저장해둔 편미분값을 모두 곱한 값
                                    : local gradient == 바로 부모에 대한 미분량. 당장 구할 수 있음
                                    Global gradient == 당장부모는 아니라 미분 바로 못해 chain rule 에 의해 구해지는 미분량
                            : 결론 )Global gradient * local gradient == 최종 업데이트할 값을 쉽게 구할 수 있다
                                : 이때 Global gradient 는 구하고자 하는 노드의 바로 이전까지 곱한 거고
                                local graident 는 구하고자 하는 노드에 저장해두었던 가중치 값인거다
                            : 이전에는 sigmoid 도 중간층에 썼었는데 더이상 안쓰는 이유는 역전파할때 chain rule 이용하기 위해
                            각 노드의 가중치를 곱하는 과정에서 1보다 작은 수가 곱해지다 보니 끝에가선 어떤 식으로 층을 쌓았던지간에 가중치가 거의 모두 0 에 수렴하여
                            갱신이 안되어 성능향상이 안됨
                                : vanishing gradient (가중치소실) 일어나는것
                                : sigmoid 그래프의 미분값은 0~0.25
                                : 딥러닝의 목표는 조금이라도 층을 깊게 쌓는것.

                    2. Sigmoid()
                        : 아웃풋레이어의 노드개수가 1개일때 activation 으로 쓰임
                        : 함수 형태) 0<= y <=1 값을 가지는 .즉 0~1 의 확률값을 가지는 .s라인 함수이다.
                        : 원리) 만약 확률값이 0.5 를 넘는다면 해당 라벨취급하고 , 그게 아니라면 그 라벨로 취급하지 않는다
                        :Dense 층에 간략화하여 들어갈때는 'sigmoid'
                    3. Softmax()
                        : 아웃풋레이어의 노드개수가 2개 이상일때 activation 으로 쓰임
                        : 원리) 각 노드별로 해당 노드를 라벨로 할 확률값을 할당하고 확률이 가장 큰 라벨로 결정
                            : 모든 뉴런의 확률값을 더했을때 1이 나와야된다
                        : 함수 형태)
                        :Dense 층에 간략화하여 들어갈때는 'softmax'로 들어감
            3. Flatten()
                :이차원이상의 데이터를 1차원화 시켜주는 인풋'레이어'
                :Dense 레이어는 1차원의 데이터만 받아들일 수 있기 때문에 필요
                :연산이 없는 레이어다
                : 형식) Flatten(input_shpae = (세로픽셀,가로픽셀,색깔여부))
                    : 색깔여부 : 3 == 컬러 . <-- 흑/백이면 걍 아예 생략

            4. Conv2D()
                :CNN (convolutin neural network ) 의 대표적인 레이어인 듯 하다
                : 특성 추출 역할을 하는 레이어
                : feature map == cnn을 통하여 특성이 추출된 이미지
                    : 하나의 cnn 에서 추출된 이미지로 이루어진 레이어는 feature maps 라고 부른다
                    : 하나의 원본에 대해 여러개의 feature maps 가 나온다.(cnn 층이 여러개다)

                : Localization ) 지역에 집중하여 특징을 추출하여 대상인식에 효과적 (Dense 처럼 모든 픽셀끼리 연산하는게 아니라 지역의 주변 픽셀들끼리 집중적으로 연산)

                :convolution 연산
                    : 쉽게생각함) 원본 = 풍경 , CNN필터(커널)= 카메라렌즈 , feature map =사진
                        : 필터 랑 커널이랑 같은 말이다.
                    : 필터를 원본에 깔고 원본과 필터의 같은 위치의 칸에 해당하는 가중치끼리 곱한 값을 모두 더하고 그 값을 feature map 의 칸에 넣는다.
                    이 과정을 필터가 원본의 끝까지 도달할때까지 진행한다.
                    : 대체적으로 feature map은 원본사이즈보다 작게 추출된다.
                        :상식적으로 생각해봐라. 원본사이즈랑 같으려면 필터사이즈가 1*1 이어야되는데 그건 의미없고 원본사이즈보다 ㅋ크려면 필터사이즈가 1*1 보다도 작아야됨
                        :사이즈가 줄어들지 않게 하고 싶으면 0 값으로 원본사진을 padding 하여 더 사이즈를 크게한 상태에서 필터를 적용한다.
                    : 한장의 필터 당 하나의 feature map 이 나오는 것이므로 하나의 대상에 대하여 여러장의 featuremap을 추출하고 싶으면 필터를 여러개 만든다.
                        :당연히 대상은 꼭 원본이 아니어도 되고 원본에 대해 추출된 이전 feature map여도 된다.
                        :당연히 각각 필터의 가중치 구성은 모두 다르다

                :형식
                    Conv2D(필터개수,필터사이즈 , activation = 'relu')
                        :해석
                            : 필터개수 == 나오는 featuremap 개수
                            : 필터사이즈 형식 ex) (3,3) --> 3*3
                                : 참고로 논문에 따르면 모든 경우에 필터사이즈는 3*3 이 효과적이라함
                                : 참고로 conv2d에 3*3 필터적용 시 가로세로 각각 -2 씩  일정하게 줄어든다
                            : activation 은 relu 고정임
                        :convolution 연산을 수행하는 레이어. 즉 cnn 레이어
                        :input 레이어에 해당하는 Conv2D레이어에는 input_shape까지 명시해준다.
                            : Conv2D(필터개수,필터사이즈 , activation = 'relu',input_shape(세로,가로,color depth))
                                : 컬러사진인경우 color depth == 3 이다. (RGB).흑백이면 1 .
                                    :color depth == 채널
                            : 후에 Dense 레이어 쓰기위해 Flatten 레이어 넣을때(Dense 레이어는 1차원만 받아들일 수 있으므로)  input_shape 또 입력하지 않는다.
                                : 존나 당연한게 input_shape 는 모델의 input 레이어에만 명시해주는 거다
                                : 그냥 빈칸으로 Flatten() 적는다.

                5. MaxPooling()
                    : 이미지의 사이즈를 줄이는 역할
                        : 그냥 원본은 사이즈가 너무 커서 픽셀값이 너무 커서 이걸 그대로 연산하기에는 연산량이 너무 크다.
                            : 이미지를 효율적으로 줄인후 연산을 해야한다
                    : Pooling 레이어의 종류
                        1. MaxPooling(한변사이즈,건너뛰는단위)
                            : 한변사이즈*한변사이즈 크기로 , 입력한 단위만큼 이동하며
                            각각의 케이스에서 최댓값 픽셀은 뽑아 이미지를 새로 만든다.
                                :한변사이즈*한변사이즈 가 conv2d의 필터의 개념인게 아니다.
                                따로 한변사이즈*한변사이즈 에 가중치가 담겨있는게 아니고 그냥 범위일 뿐이다.
                                    :헷갈릴까봐 말해주면 가중치를 뭐 곱해서 최댓값 뽑는게 아니라 그냥 해당 범위에서 원본의 최대픽셀값그냥 그대로 가져오는것
                        2. AvgPooling
                            :  요즘에는 잘 안쓴다. 평균을 내버리기  때문에 이미지가 블러리ㅣ해져서 성능이 않좋아진다.
                    : Conv2D레이어의 뒤에 온다. 꼭 모든 conv2d 레이어마다 뒤에 써줄 필요는 없다 얼만큼 자주쓸진 본인 자유
                        :ex) conv2d 마다 뒤에 써줘도 되고, 2개의 conv2d 쓰고 maxpooling 하나만 써줘도 됨

                    :한변사이즈를 n으로 설정한 maxpooling 적용하면 가로세로 각각 1/n배

                6.Dropout()
                    : 형식) Dropout(소수점)
                    : CNN 한 이후 과적합을 방지하기 위해 소수점%의 확률로 신호를 끔. 건너뜀
                        : 학습할때만 끄는거지 추론과정에선 모든 노드 반영이라고 하는데 뭔소린지 참.
                        ex) Dropout(0.25) --> 4개중 1개의 노드는 버리고 학습.
                    :Flatten() 레이어와 Dense 레이어 사이에 위치

                7. 사전학습모델변수명
                    : 형식) 그냥 말 그대로 변수명이 오는데, 젤 앞에 쓰면 된다.
                        : (이미지처리관련이었다면)그 바로 이후에는 마찬가지로 Flatten() 쓰고 Dropout 쓰고 Dense 쓰고.. 함 된다

        : model.summary() 를 쓰면 모델구성의 요약본(?) 을 볼 수 있다
            : total_parameter= 뉴런의 총 개수
            : trainable_parameter = freeze 안당한 파라미터 개수
            : Non_trainable_parameter = freeze 당한 파라미터 개수
                : 그러니까 trainable=False 줘서 freeze 당한 파라미터 개수

    4. 컴파일:집 진짜 지음
        : model.compile(optimizer = '어쩌구' , loss = '저쩌구' , metrics=['어쩌구'] )
            : optimizer ) 최적화알고리즘
                1. 'sgd' : Stochastic Gradient Descent (경사하강법)
                2. 'adam' : 'sgd' 의 업그레이드 버전

            : loss) 어떤 수식으로 손실을 구할 것인가
                1. mse : 출력층에 별도로 activation 을 설정하지 않았을때
                    : 제곱한걸 다 더한것을 평균낸것
                2. binary_crossentropy: 출력층의 activation이 sigmoid 일떄
                3. categorical_crossentropy: 출력층의 activation이 softmax 인데 원핫인코딩 된상태일떄
                    : 원핫인코딩이 됬는지 알아보는법) 학습용y데이터 , 그러니까 학습용 라벨데이터의 값 하나만 찍어서 출력해본다.
                        ex) print(y_trian[0]) 했는데 [0,1,0,0] 같은거면 된거고 [2,3,4,0] 이나 9 뭐 이딴식이면 안된거임
                    : 원핫인코딩 시키는 법) 변수 = tensorflow.ond_hot(데이터, 라벨개수)
                        : 데이터라는게 데이터 뭉치라는게 아니라 y[0] 같은 요소단위(?) 를 의미하는거다

                4. sparse_categorical_crossentropy: 출력층의 activation이 softmax 인데 원핫인코딩 안된상태일떄

            :metrics) 정확도 모티터링 옵션
                : 필수는 아니다
                : 'acc' 혹은 'accuracy' 를 쓰면 정확도까지 표시된다.

    5. fit(학습):x,y데이터 넣고 돌림
        :
        checkpoint_path = "파일명.ckpt"
        checkpoint = ModelCheckpoint(filepath = checkpoint_path,
                                    save_weights_only = True, # 가중치만 저장
                                    save_best_only = True, # 저장할 값중 가장 best 값만 저장
                                    monitor = 'val_loss', # 뭘 기준으로 할것인가
                                    verbose = 1)
        model.fit(학습용인풋데이터,
                    validation_data = (검증용인풋데이터),
                    epochs = 학습횟수,
                    callbacks= [checkpoint], # 만약에 checkpoint 필요 없으면 callbacks 안쓰는 거다
                    verbose = 실행내역 출력횟수) # 만약에 checkpoint 에서 verbose 지정해줬음 fit 에선 따로 안해도 된다.

        model.load_weights(checkpoint_path)

        :에폭(epoc): 학습 횟수
            : appropriate-fitting 이 이루어지는 epoc의 회차를 찾아 일반화가 잘된 모델을 만드는게 목표이다.
                : epoc회차가 너무 진행된 상태에서는 overfitting 즉 과적합이 일어나게 된다.
                    : overfitting 이 문제가 되는 이유는 그 이름에서 찾을 수 있다. 학습환경에 너무 핏하게 학습된 나머지 조금만 그 환경이 달랒져도 오차가 커진다.

                : how) 데이터 일부는 학습에 사용하고 일부는 학습이 잘되었는지 테스트하는 용으로 분리(주어진 모든 데이터 셋을 학습에 몰빵하는게 아니라)
                        :학습오차가 아닌(loss) 실제문제와의 오차(val_loss)가 가장 적어지는 지점을 잘 찾아야된다.
                            :학습함에 따라 학습오차는 계속 줄어들지만, 실제문제와의 오차는 줄어들다가 어느순간 커진다.
                                : 실제문제와의 오차가 작아졌다 다시 커지는건 점점 과적합 상태가 되기 때문
                            :val_loss 인 이유는 validation_loss 이기 때문
                            :ModelCheckpoint 를 이용하여 val_loss가 가장 적은 회차 정보를 저장한다
                        :학습데이터와 검증용데이터를 따로따로 fit에 써야된다
                            :validation_data 가 존재하면 loss, acc 뿐 아니라 val_loss, val_acc 까지 같이 나온다
                                : 이경우 loss , acc 는 무시하고 val_loss , val_acc 만 본다.
        : ModelCheckpoint()
            : 최종적으로 val_loss 가 가장 적었던 epoc 회차 정보를 저장(val_loss 가 낮아질때마다 그 epoc 회차 정보를 저장한다)
            : 사용방법)
                1. 내용이 저장될 파일을 ckpt확장자로 생성하고 변수에 저장해둠 <--- 굳이 저장하는건 꽤 여러번 써야되서
                2. 만든 경로를 사용하여 checkpoint 정의
                3. fit 함수의 매개변수인 callbacks 에 checkpoint 를 할당
                4. checkpoint가 저장된 변수를 load_weights 를 이용해 모델에 반영

            : verbose 해서 나오는 거 해석
                Epoch 옆의 a/b : 총 b번실행 중 a번째 실행내역
                Epoch 및의 a/b : b번 학습내용 업데이트 하는데 그중 일단 a번 업데이트 했다.
                            :그러니까 b = 전체 데이터수 / batch size
                            : a는 계속 올라가고 a==b가 되면, 그러니까 b번 업데이트를 완료하면 다음 epoch 로 넘어감
                loss : 컴파일 단계에서 설정한 걸로 구한 오차임. val_loss 랑 다른거임.
                    : 학습이 진행됨에따라 지속적으로 하락함

        : 잘 됬는지 검증 원하면 model.evaluate(검증용데이터세트) 하면 됨

    *6. 예측 : 시험에서는 딱히 필요 없음. 정답 제출 시 자동으로 해줌

"""
'''
논문을 읽어야하는 이유: 시행착오를 거치지 않아도 모델쌓을때 등등 개꿀팁 get 가능
    : abstrat , 성능평가 정도만 일단 봐라. 중요 내용 다 있다

이미지분류대회(ISVRC):
    이전년도) shallow network ( 머신러닝기반. 규칙기반)
    2012 년도) Alexnet 모델이 10% 개선시킴
        :혁신    
            1. 딥러닝 알고리즘 적용
            2. cnn 처음 도입
            3. dropout 처음 도입
            4. relu 처음도입
            5. gpu 사용한 병렬학습을 처음함
        :net = 신경망기반
    2014년도) VGGNet
        :준우등모델  
        :Conv 에서 3*3 필터 쓰는게 제일 좋다      

VGG NET 논문 참고)        
    FC == fuclly connected = dense
    Conv3-64 == 필터 64개 , 필터사이즈 3 
    LRN이라는 레이어는 없어졌음
    A,B,C,D,E 버전의 각각의 개별적인 모델
        : D , E 모델을 현업에서 많이 쓴다
            : 성능) 근소한 차이로 E>D
            : D 모델 별명 ) VGG 16 <--- 레이어16개 쓰여서
            : E 모델 별명 ) VGG 19 <-- 레이어 19개쓰여서

    논문에서 에러종류
        top_1 Error :
            1000개 카테고리에서 정답일 것 같은 카테고리 1개
        top_5 Error :
            :1000개 카테고리에서 정답일 것 같은 카테고리 5개 뽑았을때 나오는 에러율
            : 보통 top_5 Error 쓴다
    n Wieght layers 
        : 해당 모델에 n개의 레이어가 쓰였다  
        : Pooling 레이어는 weight 레이어에 포함되지 않는다(카운트안된다)



'''
"""

"""
"""
RNN : 순환신경망
    : 자연어 처리분야에서 쓰임

자연어처리
:전처리파트
    : 문장은 딥러닝 모델에 바로 투입 불가하여 인코딩(숫자의 형태로 변환)해줘야되는데 그 이전에 토큰화해줘야됨
        : 토큰 == 문장을 쪼개는 단위
            :ex) word tokenization  == 단어를 단위로 문장을 쪼갬
    :순서
        1. 단어를 기준으로 토큰화
        2. 단어사전만들기
            : 토큰화하여 만들어진 각각의 단어에 숫자값 부여
        3. 문장을 단어사전을 사용하여 숫자로 표현
        4. 깍뚜기 모양되게 길이조절
            :종류
                (1) padding: 남는 공간 있음 0으로 채움
                (2) truncating: 넘는 공간 있음 자름
            : 딥러닝 모델에 들어가는 데이터의 사이즈는 모두 동일해야되기 떄문 
        
:모델링파트   
    : 자연어 처리에서 embedding , rnn 레이어 중요
Embedding layer
    : 자연어 처리 모델의 경우 반드시 들어간다
   
    :n 차원 Embedding layer == 요소가 n 개인 레이어
        : 차원이 늘어날 수록 유사도 파악에 용이하여 더 정교해짐
        :[0.1 , 0.5 , 0.3] == 3차원 Embedding layer
        
    : 기능) 단어를 Embedding vector 화 한다
        : Embedding 이란 lookup table 같은거다
            : 값을 가져올 때  굳이 인덱스로 다 순회 안하고 Key값을 이용해 빠르게 찾는게  lookup table 
        : Embedding vector ) 단어를 좌표계의 형태로 나타낸것
            :[0.1 , 0.5 , 0.3] 같은거
            
    : Embedding layer 필요한이유) 단순히 단어를 숫자화 하는 것을 넘어 단어간의 유사도, 관계를 파악할 수 있게 해준다
        : 원리) n차원 공간에 해당 좌표에 해당하는 위치에 그 단어를 꽂고 다른 단어들 간의 거리를 이용하여 유사도 계산
        : 단순히 각 단어에 정수값을 부여하여 단어사전을 만들면 단어1 + 단어2 = 단어7 과 같은 전혀 상관없는 관계를 형성할 수 ㅇ있기 때문
        : 원핫인코딩을 사용하면 각각의 단어를 전부 독립적이게 만들기 때문에 관계 파악이 어려움. 그리고 거의 대부분의 공간ㄴ이 0으로 채워져있어 비효율적 연산
            : 0으로 대따 많이 채워져 있는 이런걸 sparse vector 라고 한다       
    : 사용방법) Embedding(단어사전커트라인크기 , 임베딩벡터차원 , input_length = 문장디폴트길이 )

RNN (= Recurrent Neural Network) : 순환신경망
    : 딥러닝의 꽃
    : 데이터의 순서를 고려하여 순차적으로 학습하는 레이어      
    
     :Valilla RNN
        : 가장 순정 rnn 모델. 꾸밈없는 모델
        : cell 
            : 각 포인트를 노드가 아니라 셀이라고 한다
            : 구성
                1. tanh 
                    : activation 이다.
                    : sigmoid 랑 유사하게 생겼는데 미분량의 범위가 -1~1로 그래프가 더 가파르다
                        : sigmoid 경우 0~0.25 라서 깊게 쌓을 경우 vanishnig gradient 일어날 위험있다
                        : sigmoid 에 비해서 vanishnig gradient 일어날 확률적다
                2. 수식( W*X + b 같은)
            : 토큰화 해서 만들어진거 개수만큼 만들어진다
                : 토큰화된 대상이 각각 셀과 대응되어 만들어진다
                : 그러니까 단어를 기준으로 토큰화 했음 단어의 개수만큼 만들어진다 
         : 원리)
            들어온 0번째 단어에 대해  0번째 셀에서 학습된 가중치를 1번째 셀로 전달한다
            '앞에서 들어온 가중치' + '새로 들어온 1번째 단어에 대해 새로 학습된 가중치 ' 를 1번째 셀의 가중치로 한다
            1번째 셀의 가중치를 2번째 셀로 넘기고 이와 같은 과정을 반복한다.
        : 문제점) 문장이 길면 오차역전파의 chain rule 적용 과정에서 연산이(곱하기가) 너무 많이 일어나 vaninshing gradient 발생
        : 해결) LSTM
        
    :LSTM(long short trem memory) 레이어
        : vanishing gradient 를 방지하기 위한 대책.
        : vanilla model 에 선하나 추가 한 구조
            : 이전 셀에서 다음 셀로 넘어갈때 연결되는 선이 2개가 된거다
            : 선 하나는 Longterm 메모리, 나머지 하나는 shortterm 메모리 관리
                : 중요도가 큰 단어라고 판단되면 Longterm 메모리에 올려 끝까지 보존 될 수 있게 함
                
        :구성) 참고다.참고. 굳이 알 필요없다
            1. forget gate: 어느정도로 잊을지 결정    
            해당 셀에 새로 들어온 데이터에 대한 가중치와 이전 셀에서 들어온 가중치를 합한것에
            sigmoid를 곱한다.
                :  sigmoid가 0 에 가까운 값이면 불필요한 정보이기에 거의 반영 안되게 한것
                :  sigmoid가 1 에 가까운 값이면 필요한 정보이기에 장기기억에 반영되게 하는것
            2. input gate:
                : forgetgate 와 같은 수식 사용
            3. cellstate: 장기깅억 관련 처리
                : forget gate 비스무리 수식을 거치고 이전 셀의 sigmoid 값과 현재 셀의 tanh 값을 곱한다.
                    : 마이너스 값이 나올 수 있기에 방향성이 ㄱ결정된다.        
                    : 방향 따라서 장기기억에 반영시킨다
            4. outputgate 
                : forget gate와 같은 수식
            5. hiddenstate: 단기기억관련 처리
                : 장기기억으로부터 흘러온 정보와의 곱셈연산을 통하여 최종 아웃풋결정
        
        :LSTM 레이어 사용해 할 수 있는 모델링
            :종류
            1. one to one : 입력되는 데이터 하나, 출력되는 데이터 하나
            2. one to many: 입력되는 데이터 하나, 출력되는 데이터 여러개
                ex) 사진 보고 느낀 감정들 서술
            3. many to one : 입력되는 데이터 여러개 , 출력되는 데이터 하나
                : 4번 유형 문제
                :ex) 문장 투입후 sarcastic 한지 판단
            4. many to many: 
                            :종류
                                (1) 일대일 매치
                                    : 요즘에 잘 쓰임 
                                    : 5번 유형 문제
                                (2) 일대일 매치는 아니고 일련의 과정을 거쳐 해석한 이후 다시 풀어서 제출
                                    : 과거 번역모델에 잘 쓰였었음
                                        ex) I love you --> 단어 단위로만 번역하면 한국어 어순에 맞지 않는데, 이를 사랑한다는 의미를 중심으로 한국어로 풀어 출력          
           
            :Many to one 과 Many to many 의 비교
                    :최종 레이어 코딩부분)
                        : 겹쳐안쓰면 첫레이어를 말하는거고 겹쳐쓰면 마지막레이어를 말하는거임
                        :Many to one --> LSTM(n)
                            : n == Conv2D 에서의 필터개수와 같은 역할
                                : 연산 역할을 하는게 아니라 특성 추출의 역할을 하는거임
                                : n개의 특성값을 추출한다는거
                                
                        :Many to Many --> LSTM(n , return_sequences = True)
                            : 각 시퀀스 별로 리턴값을 주기 때문에 리턴값이 여러개 나온다
                            : 물론 n == Conv2D 에서의 필터개수와 같은 역할
                    
                    : Stacking LSTM
                        : LSTM 의 층을 여러개 겹쳐서 사용하는 행위
                        : 다음레이어를 LSTM 으로 받는 LSTM레이어의 경우 무조건 return_sequences = True 해줘야된다
                            : 다음층이 LSTM 이라 치면 many로 받아야될테니 이전 층에서 output 도 여러개 나와야되서
                            : output 레이어는 상황에 따라서 return_sequences = True 한다
                                : many to one 이면 안쓰는거고 many to many 면 쓰는거고
            

Bidirectional 레이어
    : 양방향으로 특성추출하는 레이어
        : 때로는 뒷방향으로 특성추출할때가 유리하기도 해서
            ex) 나는 XX를 뒤집어 쓰고 울었다 에서 '나는 XX' 보다 'XX를 뒤집어쓰고 울었다' 가 더 알기 쉬움
    :Bidirectional(LSTM(n)) --> 앞에서 n개 , 뒤에서 n개의 특성추출한다. 그러니까 총 2*n개의 특성이 추출된다.
"""
'''
참고로 일차원 데이터를 벡터라고 한다
    :[12,3,4,5] 같은
'''
"""
시계열 데이터에서의 전처리
    : 사실 좀 복잡하다. 입문자 수준에서는 도달 어렵다. 문제에서 주어준다
    
시계열 데이터
    : 계절성 , 주기 , 트렌드를 모두 포함하면서 시간의 흐름대로 되어있는(순차적인) 데이터
        ex) 스키장사 관련 데이터
시퀀스(Sequence)
    : 계절성 , 주기 , 트렌드를 포함하지 않으면서 시간의 흐름대로 되어있는(순차적인) 데이터
        ex) 주가 <-- 경제 상황에 따라서 움직이지 딱히 계절의 영향을 받지 않음
    : 태양의 흑점데이터는 시퀀스이다
    : 사실 시퀀스, 시계열 퉁쳐서 별 구분없이 부르기도 한다
    

       
windowed dataset
    : 시계열 데이텉 다룰 때 꼭 나오는 개념
    : 활용방법
        def windowed_dataset(series , window_size , batch_size , shuffle_buffer):
            series = tf.expand_dims(series , axis = -1) # 2차원화
            ds = tf.data.Dataset.from_tensor_slices(series) # 2차원의 데이터를 Dataset 클래스화
            ds = ds.window(x,y데이터를 모두 포함하는 사이즈 , shift = n, drop_remainder = True) #windowed dataset 구성
            ds = ds.flat_map(lambda w: w.batch(x,y데이터를 모두 포함하는 사이즈))) #평탄화 , batch 사이즈 설정
            ds = ds.shuffle(shuffle_buffer) #셔플함
            ds = ds.map(lambda w: (w[슬라이싱] , w[슬라이싱])) #x,y 데이터 분할
                return ds.batch(batch_size).prefetch(1)          
            : series 매개변수 = 일련의 데이터를 받는다
                :[1,2,3,4,5] 같은거
            : expend_dims( series , axis = -1)
                : 차원을 늘리는 역할
                :  window_dataset 트ㄱ성상 2차원의 데이터를 받아야되서 2차원화 시켜줘야됨
                    : 만약에  series 자체적으로 2차원인 경우 쓸 필요 없지만 1차원의 데이터를 받는경우 반드시 써줘야
                    : ex)[1,2,3,4,5] -->[[1],[2],[3],[4],[5]] 
                : -1 == 제일 끝부분
            : from_tensor_slices(series)
                : Dataset 이라는 클래스로 변환시켜준다
                    : 그래야 추후 유용한 함수들(window , shuffle , flat ..) 을 사용할 수 있게되기 때문
            : window(x,y데이터를 모두 포함하는 사이즈 , shift = b, drop_remainder = True)
                :windowed dataset 구성
                : x,y데이터를 모두 포함해야하는 사이즈ex) 
                    :[0,1,2,3,4,5] 에서 x데이터 =[0,1], y데이터 = [2,3] 이런식이라 하면 4 로 설정해야
                        
                : shift 는 상황봐가면서 조절. drop_remainder 는 True 고정
            : flat_map(lambda w: w.batch(x,y데이터를 모두 포함하는 사이즈)))
                : Flatten  기능 + mapping 기능
                    : 평탄화시켜주면서 간단한 함수를 적용시킬 수 있게됨            
                : 앞서서 2차원으로 만들어 진행했으므로 1차원화 시켜줘야됨
                : w가 큰 의미가 있는게 아니라 그냥 앞선 레이어의 가중치를 받는 매개변수임
                
            :shuffle(shuffle_buffer)
                :shuffle_buffer 사이즈만큼 셔플함
                
            :map(lambda w: (w[슬라이싱] , w[슬라이싱]))
                : x, y 데이터 분할
                    : 앞의 w[슬라이싱] 은 x 데이터 , 뒤의 w[슬라이싱] 은 y데이터 부분된다
                : ex) [0,1,2,3,4,5] 에서 (w[:-1] ,w[1:])
                    :x데이터 = [0,1,2,3,4] <--- -1번째요소 즉 맨끝 요소 전까지
                    :y데이터 = [1,2,3,4,5]
            :batch(batch_size).prefetch(n) 
                :windowed dataset 은 동적 데이터셋이다
                    : 그러니까 학습 들어가기전에 이미 만들어진 데이터(정적데이터) 가 아니라 학습이 진행됨에 따라 생기는 데이터인데
                    메모리를 효율적으로 쓸 수 있다는 장점이 있지만 병목현상이 일어난다는 단점이 있다
                    : 그러니까 한번의 배치만큼 학습이 끝나고 나서야 배치를 만들고, 그 배치로 학습한 후 또 배치를 만들어 학습하고 ... 이런식으로 진행되어 시간이 좀 오래걸린다.
                : 병목현상 방지를 위하여 미리 n 개의 배치를 만들어 두겠단 의미 
                    : 학습속도가 더 개선됨
    : 시퀀스 데이터를 활용해 windowed dataset 만드는 법
        : 특정 기준으로 묶어주어야한다 .
        : 구성
            : window_size = 묶는 사이즈
                : 무조건 길다고 좋은것도 아니고 작다고 좋은게 아니라 적젛라게 알아서  해야됨
                : 하이퍼파라미터이다
            : x,y 데이터 ) window_size 만큼 묶어지면서 문제 상황에 따라 정해짐
                : 이때의 y 데이터를 타겟 데이터라고도 한다
                : many to one 도 가능하고 many to many 도 가능하다
            : shift: 몇칸씩 이동하며 window_size 만큼 묶어 x,y데이터를 설정할 것인가
            : drop_remainder
                : 반드시 True 로 지정해준다
                    : 만들 수 있는데까지만 만든다(일정한 크기의 깍뚜기 데이터 만든다)
                : False 로 지정되있음 남아있는게 window_size 만큼 되지 않아도 shift 하면서 끝까지 건들이기 때문에(?)
                 데이터가 에쁜 깍뚜기 모양으로 안나온다
                    :ex) [0,1,2,3,4] 에서 window_size = 3 ,shift =1 인데 drop_remainder =False 면 
                        [0,1,2],[1,2,3], [2,3,4] , [3,4] , [4] 로 일정하지 않은 데이터가 만들어진다
            : shuffle_buffer : 버퍼사이즈를 설정하는것  
                : 설정종류
                (1) full shuffle : 처음부터 끝까지 다 섞는것
                    : 데이터의 개수만큼 버퍼사이즈를 설정하면됨
                    : 잘 사용안함
                        : 메모리가 많이 낭비되고 불필요한 연산이 일어나서
                
                (2) 데이터의 개수보다 작게 버퍼사이즈를 설정하는것
                    : 잘사용함.
                        :메모리도 절약적이고 시간측면에서 ㅇ유리하기에 버퍼를 많이 사용
                    : 원리)
                        데이터에서 버퍼사이즈개수만큼 랜덤하게 뽑아와서 섞어 준비해둠
                        모델이 학습할때 버퍼에게 n개의 데이터를 요구하고 버퍼가 n개의 데이터를 빼서 모델에게 제공함.
                        n개만큼의 데이터가 빠졌음으로 데이터에서 n개의 데이터를 랜덤하게 또 뽑아와 보충함
                            : ex) a개의 기존데이터가 있을때 버퍼사이즈가 b 개이고 모델은 n개의 데이터를 요구
                                1. 버퍼가 b개의 데이터를 뽑아서 내부적으로 섞어둔다 
                                    : 기존데이터 = a-b 개 , 버퍼에 담긴 데이터 = b개
                                2. 모델이 n 개의 학습데이터를 요구하여 버퍼에서 n개 빼가고 버퍼는 기존데이터에서 n개만큼 뽑아와 보충
                                    : 기존데이터 = a-b-n개 , 버퍼에 담긴 데이터 = b개
                                      
        :ex) [0,1,2,3,4,5,6,7,8,9] 의 시퀀스 데이터
            : 과거 3일치의 데이터로 다음 하루칭의 값을 예상하고 싶다
                : window_size = 3
                : shift =1
                : x데이터 = [0,1,2],[1,2,3] ..., 
                : y데이터 =3 , 4, 5...
                    : 이 케이스에서 x데이터는 many, y 데이터는 one 
               
            
 모델링:
 conv1D 사용:
    :conv2D보다 훨씬 간단
    : 형식)
        Conv1D(특성값개수 , kernel_size = a,
                padding = '어쩌구' ,
                activation = 'relu',
                input_shape=(None,예측값개수))

        :kernal_size
            : 시퀀스 데이터에서 얼마의 사이즈로 convolution 연산을 해서 특성값을 뽑아낼 건지 결정
        : strides 
            :kernal_size 를 얼마 간격으로 이동하면서 적용할건지       
        : padding
            : convolution 연산을 수행했을때 나오는 값들의 개수가 실제 개수보다 작아서 앞의 일부 값들은 맵핑 못당하고 버려지는데
            이를 방지하기 위하여 패딩값을 앞부분에 넣어 적용해도 사이즈가 줄어들지 않게 된다
            : 'casual' 로 설정하면 0로된 패딩값을 넣는다
        : input_shape
            :() 쓰든 [] 쓰든 상관 없다 
            :None 쓰면 window_size 가 자동으로 들어간다
                : 그러니까 그냥 window_size 쓴거랑 마찬가지
            :예측값 개수 ex) 흑점활동 하나 예측 --> 1 쓴다. 저녁,점심,아침메뉴예측--> 3쓴다
             
    :  conv1D 사용결과를 LSTM 에 넣고 쭉 모델링을 쌓는다

Momentum(=관성):
    :sgd 보다 adam이 좋은 이유
        :Momentum 값을 조정함으로 개선했으니까
            : 손실함수는 실제로는 매끄러운 곡선 함수가 아니라 울퉁불퉁한 곡선함수.
                :목표는 최솟값이 되는 지점을 찾는거임(극솟값 지점이 아니라)
                :Momentum(=관성) 이 zero 로 되있으면 마찰력이 매우 높단 거임
                    : 굴곡을 넘어서지 못하고 극솟값에 빠져버림
                :Momentum 을 0.9 정도로 설정하면 관성을 받아서 ,마찰력이 줄여져서,가속도가 붙는다 
                    : 언덕에서 내려올때 일정한 속도로 가는게 아니라 내려감에 따라 가속도(가중치)가 붙어 언덕을 넘을 수 있게되어 
                    극소지점에 고립되지 않게 됨
옵티마이져의 학습률 바꾸기
    : 변수 = tf.keras.optimizers.옵티마이져명(learning_rate = 원하는값 , momentum = 원하는값) 하고
    변수를 옵티마이져로써 쓰면됨
        : 그러니까 model.compile 부분에서 'optimizer = 변수 ' 로 하면됨               

Huber Loss
    :sequentail 데이터 학습시킬떄 많이 사용
    :Absolute error 와 MSE 섞은것.
        : 특정지점에서는 둥근모양 그 부분제외지점에선 껶인모양
        Squared error(MSE): 제곱값 --> 둥근 모양
        Absolute error : |오차값| -- >  꺾인 모양
        
"""

"""
이미지: 픽셀들로 이루어짐
    : 픽셀
        : 값은 0~255 의 숫자를 가짐
    
    : 정규화(Normalization)해서 사용
        : 픽셀들이 0~1 사이의 값을 가지게 하여 성능을 높이는 것 
            : 숫자의 범위가 0~255 로 데이터의 범주가 넓으면 큰폭으로 그래프가 그려저 미분값이 어떤땐 매우 크고 매우 작게 나오는데(불안정하지만) 
            0~1 로 좁으면 비교적 안정적인 접근이 가능
        
        : 픽셀 = 픽셀/255 를 하여 실행
            : 픽셀 = 0~255 값을 가지니까
        
        : 이미지 쓸때 거의 무조건 함.
"""
"""
원핫인코딩: 하나만 핫하다
    : 디코딩은 숫자를 문자열로 바꾸는 것
    : 필요한 이유 :딥러닝 모델은 숫자만을 받을 수 있기 때문에 문자열을 숫자화 해야됨
    : 각각의 라벨에 단순히 0,1,2,3.. 이런식으로 값을 할당했을떄 1+2 = 3 과 같은 연산을 적용하여 라벨들간의 말도 안되는 관계가 형성될 수 있음
    : 형태) [0,1,0,0,0]
        : 요소의 개수 == 라벨의 개수
        : 해당하는 라벨의 인덱스에 1 로 표시, 나머지는 모두 0
    : 원핫인코딩을 통하여 만든 라벨들은 독립적이라고 한다.
        : 그렇다고 한다..    
"""
'''
그러니까 복잡도 정리해보면

LV0: 입력층만 달랑 존재(라벨링의 필요가 없음)
LV1: 입력층과 출력층이 존재(랍벨링필요)
LV2: 히든레이어존재-activation 없이
    : 어느정도 복잡하니까 modelcheckpoint 로 val_loss 최하지점 캐치
LV3: 히든레이어존재-activation 있게
    : 복잡하니까 modelcheckpoint 로 val_loss 최하지점 캐치
'''