
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import cv2
import shutil
from tensorflow import keras



# ======================================================================================
# Variables
IMG_SIZE = 227

TEST_CSV = './test_vision.csv'
IMG_PATH = './faces_images/faces_images/'

MODEL_PATH = './models/'

tf.executing_eagerly()
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_HEIGHT = 227
IMG_WIDTH = 227

# ---------------------------------------------------------------------------------------------------------------------

def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as ex:
        print(str(ex))



def ALEX_NET(NUM_CLASSES):
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
    dense3 = keras.layers.Dense(NUM_CLASSES, activation='softmax')(drop2)
    return keras.Model(inputs=inputs, outputs=dense3)


# ========================================================================================
# Load Models

PROJECT = 'all'
LABEL = ['1', '2', '3', '4', '5', '6']

model = ALEX_NET(len(LABEL))
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=5e-5, momentum=0.9)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.load_weights('{}{}.h5'.format(MODEL_PATH, PROJECT))


# ============================================================================================
# Test

test_csv = pd.read_csv(TEST_CSV, header=0)
test_filename = test_csv['filename']

mkdir('result')


for img_src in test_filename:
    img = cv2.imread(IMG_PATH + img_src)
    img = cv2.resize(img, (227, 227))
    img = img/255.0
    img = tf.dtypes.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [1, IMG_SIZE, IMG_SIZE, 3])
    predictions = model.predict(img)
    p_val = np.argmax(predictions[0])       

	label = str(p_val +1)
    mkdir('result/' + label)        
    shutil.copy(IMG_PATH + img_src, 'result/' + label + img_src)



# =========================================================================================



