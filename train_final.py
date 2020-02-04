
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow import keras
import pandas as pd
import shutil


# ==============================================================================================
# Variables

NUM_CLASSES = 6
IMG_SIZE = 128                                      # IMG_SIZE
IMG_PATH = './datset/faces_images/'
TRAIN_CSV = './dataset/train_vision.csv'
TEST_CSV = './datset/test_vision.csv'
FILE_PATH = './dataset/'                            # Data folder
TRAIN_PATH_1 = FILE_PATH + 'train/1/'            # class 1 이미지
#VALID_PATH_1 = FILE_PATH + 'valid/1/'
TRAIN_PATH_2 = FILE_PATH + 'train/2/'          # class 2 이미지
#VALID_PATH_2 = FILE_PATH + 'valid/2/'
TRAIN_PATH_3 = FILE_PATH + 'train/3/'                 # class 3 이미지
#VALID_PATH_3 = FILE_PATH + 'valid/3/'
TRAIN_PATH_4 = FILE_PATH + 'train/4/'            # class 4 이미지
#VALID_PATH_4 = FILE_PATH + 'valid/4/'
TRAIN_PATH_5 = FILE_PATH + 'train/5/'          # class 5 이미지
#VALID_PATH_5 = FIUTILLE_PATH + 'valid/5/'
TRAIN_PATH_6 = FILE_PATH + 'train/6/'                 # class 6 이미지
#VALID_PATH_6 = FILE_PATH + 'valid/6/'


tf.executing_eagerly()
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
IMG_HEIGHT = 227
IMG_WIDTH = 227
epochs = 150



# ==============================================================================================
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('already exists', path)
  



# ===============================================================================================
# Loading Data
# train/test image 분류
SHUFFLE = True
VALID_PER = 0.1


train_csv = pd.read_csv(TRAIN_CSV, header=0)
#test_csv = pd.read_csv(TEST_CSV, header=0)

train_data = []
for train in train_csv.iterrows():
    train_data.append([train[1]['filename'], train[1]['label']])

#test_filename = test_csv['filename']

mkdir(TRAIN_PATH_1)
mkdir(TRAIN_PATH_2)
mkdir(TRAIN_PATH_3)
mkdir(TRAIN_PATH_4)
mkdir(TRAIN_PATH_5)
mkdir(TRAIN_PATH_6)
mkdir('model')
#util.mkdir(TEST_PATH)




# ----------------------------------------------------------------------------------------------
# train data 1/2/3/4/5/6 분류
files_1 = []
files_2 = []
files_3 = []
files_4 = []
files_5 = []
files_6 = []
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

#for testfile in test_filename:
#    shutil.copy(IMG_PATH + testfile, TEST_PATH + testfile)


print('1 images: ', len(files_1), '2 images: ', len(files_2), '3 images: ', len(files_3))
print('4 images: ', len(files_4), '5 images: ', len(files_5), '6 images: ', len(files_6))


# -----------------------------------------------------------

#
train_1 = files_1
for train in train_1:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_1 + train + '.png')
#
train_2 = files_2
for train in train_2:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_2 + train + '.png')
#
train_3 = files_3
for train in train_3:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_3 + train + '.png')
#
train_4 = files_4
for train in train_4:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_4 + train + '.png')
#
train_5 = files_5
for train in train_5:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_5 + train + '.png')
#
train_6 = files_6
for train in train_6:
    shutil.copy(IMG_PATH + train + '.png', TRAIN_PATH_6 + train + '.png')

print('train 1: ', len(train_1), 'train 2: ',len(train_2), 'train 3: ',len(train_3))
print('train 4: ', len(train_4), 'train 5: ',len(train_5), 'train 6: ',len(train_6))

train_dir = pathlib.Path('./dataset/train')
#valid_dir = pathlib.Path('./datasets/validation')
total_train_data = len(list(train_dir.glob('*/*.png')))
#total_val_data = len(list(valid_dir.glob('*/*.png')))
print(total_train_data)
#print(total_val_data)

CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])

def NETWORK():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    conv1 = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='same',
                                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                activation='relu')(inputs)

    conv2 = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')(conv1)
    norm1 = tf.nn.local_response_normalization(conv2)
    pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm1)

    conv3 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    norm2 = tf.nn.local_response_normalization(conv3)
    pool2 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm2)

    conv4 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    conv5 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
    pool3 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv5)

    flat = keras.layers.Flatten()(pool3)
    dense1 = keras.layers.Dense(4096, activation='relu')(flat)
    drop1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(4096, activation='relu')(drop1)
    drop2 = keras.layers.Dropout(0.5)(dense2)
    dense3 = keras.layers.Dense(6, activation='softmax')(drop2)
    return keras.Model(inputs=inputs, outputs=dense3)


model = NETWORK()
model.summary()


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=5e-5, momentum=0.9)
model.compile(
    # optimizer='adam',
    optimizer=optimizer,
    # loss='binary_crossentropy'
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                                                                        rescale=1./255,
                                                                        rotation_range=10,
                                                                        horizontal_flip=True,
                                                                        zoom_range=0.5,
                                                                        shear_range=0.2
)

#valid_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_image_generator.flow_from_directory(
    directory=train_dir,
    # resize train data
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
)

#valid_generator = valid_image_generator.flow_from_directory(
#    directory=valid_dir,
#    target_size=(IMG_HEIGHT, IMG_WIDTH),
#    batch_size=BATCH_SIZE,
#    class_mode='categorical'
#)

start_time = 'face' + datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

log_dir = './model/logs/' + start_time
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=150, verbose=1)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=total_train_data//BATCH_SIZE,
    epochs=epochs,
    verbose=1,
    #validation_data=valid_generator,
    #validation_steps=total_val_data//BATCH_SIZE,
    callbacks=[early_stopping_callback, tensorboard_callback]
)

model.save(log_dir+'/'+start_time+'.h5')
