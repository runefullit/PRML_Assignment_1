'''
DATA.ML.200-2020-2021-1 Pattern Recognition and Machine Learning
Assignment 1 - Afro-MNIST
Olli Lähde & Aleksi Mäki-Penttilä
H263200
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist
from os import listdir
from os.path import dirname
from PIL import Image as PImage
import numpy as np
import matplotlib.pyplot as plt


def load_afro_MNIST():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(1, 11):
        print(f'Working on training set {i}')
        path = f"{dirname(__file__)}/train/{i}/"
        imagesList = listdir(path)
        cutoff = 5000
        j = 0
        for image in imagesList:
            img = np.asarray(PImage.open(path + image))
            if j < cutoff:
                x_train.append(img)
                y_train.append(i-1)
            else:
                x_test.append(img)
                y_test.append(i-1)
            j+=1
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

(x_train, y_train), (x_test, y_test) = load_afro_MNIST()

# Normalizing the image data.
x_train = x_train / 255.0
x_train = x_train[...,None]
x_test = x_test / 255.0
x_test = x_test[...,None]

#import sys
#sys.exit()

model = keras.Sequential(
    [
        # Preprosessing this already helps with overfitting since the
        # training dataset is now arbitrarily larger and more varied.
        #layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(28,28, 1)),
        #layers.experimental.preprocessing.RandomRotation(0.15),

        layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax")
    ]
)
model.summary()

# Used to exit program in order to inspect the model before training.
#import sys
#sys.exit()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(x_train, y_train,  epochs=10, batch_size=64, verbose=2)
loss, accuracy  = model.evaluate(x_test, y_test,  verbose=2)


print(f'Network Accuracy: {round(accuracy*100, 2)}%')