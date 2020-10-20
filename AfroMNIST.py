'''
DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning
Exercise 4 - Neural networks using Tensorflow
Olli Lähde
H263200
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist
import numpy as np



(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalizing the image data.
x_train = x_train / 255.0
x_train = x_train[...,None]
x_test = x_test / 255.0
x_test = x_test[...,None]

# Defining mish activation function, online sources were adamant
# that this would provide better results than relu which would
# provide better results than sigmoid although I don't fully
# understand how these activation functions make their difference.
# Using mish instead of relu seems to increase accuracy very slightly
# but individual epoch times rise ~20%.
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


model = keras.Sequential(
    [
        # Preprosessing this already helps with overfitting since the
        # training dataset is now arbitrarily larger and more varied.
        #layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(28,28, 1)),
        #layers.experimental.preprocessing.RandomRotation(0.15),

        layers.Conv2D(32, (3,3), activation=mish, input_shape=(28,28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation=mish),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation=mish),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation=mish),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        
        layers.Dense(512, activation=mish),
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