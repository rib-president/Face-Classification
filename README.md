
# **Face-Classification-by-Gender/Age/Glasses**
  
  

## Table of Contents  
* [About](#about)
* [How to](#how-to)
* [Problem-Solving](#problem-solving)
* [Architecture](#architecture)
* [Result](#result)



## About
성별/연령/안경 착용 여부에 따라  
  

* class 1: plain male  
* class 2: glasses male  
* class 3: boy  
* class 4: plain female  
* class 5: glasses female  
* class 6: girl  
  
총 6개의 클래스로 인물 사진을 분류합니다.  
**학습 데이터**는 무작위로 크롤링 된 인물사진들과 [이미지 파일명, class]로 이루어진 csv 파일입니다.  
  
  
|filename|label|
|:---:|:---:|
|0001.jpg|3|
|0002.jpg|5|
  


## How to
### 1. Data Preprocessing
원활한 학습을 위해 학습 이미지를 클래스 별로 나누어야 함. 클래스 정보가 있는 csv 파일을 읽어 이미지를 알맞은 폴더 이동  
    
  
``` python
train_csv = pd.read_csv(TRAIN_CSV, header=0)
#
train_data = []
for train in train_csv.iterrows():
    train_data.append([train[1]['filename'], train[1]['label']])
#
# ------------------------------------------------------------------
for trainfile, label in train_data:
    if label == 1:
        files_1.append(trainfile[:-4])
    elif label == 2:
        files_2.append(trainfile[:-4])
    elif label == 3:
        files_3.append(trainfile[:-4])
    elif label == 4:
        files_4.append(trainfile[:-4])
    elif label == 5:
        files_5.append(trainfile[:-4])
    elif label == 6:
        files_6.append(trainfile[:-4])  
#
# ---------------------------------------------------------------------
for train in files_1:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_1 + train + '.png')
#
for train in files_2:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_2 + train + '.png')
#
for train in files_3:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_3 + train + '.png')
#
for train in files_4:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_4 + train + '.png')
#
for train in files_5:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_5 + train + '.png')
#
for train in files_6:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_6 + train + '.png')
```
  

> augmentation은 학습 단계에서 keras ImageDataGenerator로 진행
  
``` python
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
     rescale=1./255,
     rotation_range=10,
     horizontal_flip=True,
     zoom_range=0.5,
     shear_range=0.2
)
#
train_generator = train_image_generator.flow_from_directory(
    directory=train_dir,
    # resize train data
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
)

```
  
  
### 2. Modeling
dataset이 작고 시간적 여유가 없는 상태에서 pretrained model을 사용할 수 없는 조건 이었기에 ALEXNET으로 구현
* Conv Layer
  * 2D Conv + ReLU, local_response_normalization, Max-pooling
  
* FCN Layer
  * Dense + ReLU, dropout
  
  
``` python
def NETWORK():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    # -----------------------------------------------------------------------------------------
    conv1 = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='same',
                                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                activation='relu')(inputs)
    # -----------------------------------------------------------------------------------------
    conv2 = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')(conv1)
    norm1 = tf.nn.local_response_normalization(conv2)
    pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm1)
    # -----------------------------------------------------------------------------------------
    conv3 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    norm2 = tf.nn.local_response_normalization(conv3)
    pool2 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm2)
    # -----------------------------------------------------------------------------------------
    conv4 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    conv5 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
    pool3 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv5)
    #
    # -----------------------------------------------------------------------------------------
    flat = keras.layers.Flatten()(pool3)
    dense1 = keras.layers.Dense(4096, activation='relu')(flat)
    drop1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(4096, activation='relu')(drop1)
    drop2 = keras.layers.Dropout(0.5)(dense2)
    dense3 = keras.layers.Dense(6, activation='softmax')(drop2)
    return keras.Model(inputs=inputs, outputs=dense3)

```
  

### 3. Inference
test 이미지가 model이 분류한 label 폴더에 알맞게 저장
  

``` python
for img_src in test_filename:
    img = cv2.imread(IMG_PATH + img_src)
    #
    img = cv2.resize(img, (227, 227))
    img = img/255.0
    img = tf.dtypes.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [1, IMG_SIZE, IMG_SIZE, 3])
    #
    predictions = model.predict(img)
    p_val = np.argmax(predictions[0])
    #
    label = str(p_val +1)
    mkdir('result/' + label)
    #
    shutil.copy(IMG_PATH + img_src, 'result/' + label + img_src)
```
    
  

## Problem-Solving
프로젝트 초기에는 통으로 6개의 class를 구별하는 것보단, 공통점이 있는 class끼리 두 번의 stage를 거쳐 inference 되게 하였습니다.  
* solution 1  
![](https://github.com/rib-president/Face-Classification/blob/master/sample/face_solution_1.JPG/)


* solution 2
![](https://github.com/rib-president/Face-Classification/blob/master/sample/face_solution_2.JPG/)
  
  
* solution 3
![](https://github.com/rib-president/Face-Classification/blob/master/sample/face_solution_3.JPG/)
  

하지만 여러 번의 학습-test 결과 현재의 6 class model 보다는 모두 정확도가 높지 않았습니다.  
label별 이미지의 개수가 천차만별이었고, dataset 또한 제한적이었어서 이러한 결과가 나온 것으로 추측됩니다.  
> keras ImageDataGenerator 이외에 추가적인 augmentation도 진행하였으나, 원체 원본 데이터가 적은 label 이미지가 있어 정확도가 크게 향상되지는 않았습니다.  
  


## Architecture  
  
|Stages|Preprocessing|Modeling|
|:----:|:-----------:|:------:|
|Libraries|pandas, keras|keras, tensorflow|
Language|Python|Python|
  
  
## Result
#### <center> class 3: boy로 분류된 이미지들</center>
![](https://github.com/rib-president/Face-Classification/blob/master/sample/face_3.JPG/)

  
#### <center> class 4: plain female로 분류된 이미지들</center>
![](https://github.com/rib-president/Face-Classification/blob/master/sample/face_4.JPG/)

  
#### <center> class 2: glasses male로 분류된 이미지들</center>
![](https://github.com/rib-president/Face-Classification/blob/master/sample/face_2.JPG/)

  
  
