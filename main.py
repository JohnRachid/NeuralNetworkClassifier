import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    # data is 64 + number which represents the given data's number so its 8 x 8
    # input matrix of 8x8 where each element is an integer in the range 0..16  + last value which is the number

    testing_df = pd.read_csv('data/test/optdigits.tes', header=None)
    X_testing, y_testing = testing_df.loc[:, 0:63], testing_df.loc[:, 64]

    training_df = pd.read_csv('data/train/optdigits.tra', header=None)
    X_training, y_training = training_df.loc[:, 0:63], training_df.loc[:, 64]

    X_train = X_training.to_numpy().reshape(-1, 8, 8, 1)
    X_test = X_testing.to_numpy().reshape(-1, 8, 8, 1)

    y_train = keras.utils.to_categorical(y_training, 10)
    y_test = keras.utils.to_categorical(y_testing, 10)

    # for i in range(9):
    #     plt.subplot(331+i)
    #     plt.imshow(X_train.reshape(-1,1,8,8)[i][0])
    # plt.show()
    # # print(y_test[1000:1009])

    print(X_train.shape)
    print(y_train.shape)
    trainModels(2, 25, 64, 128, X_train, y_train, X_test, y_test)


def trainModels(numHiddenLayers, numEpochs, numHiddenUnitsPerLayer, batchSize, X_train, y_train, X_test, y_test):
    # model = tf.keras.Sequential([
    #     # Adds a densely-connected layer with 64 units to the model:
    #     layers.Dense(64, activation='relu', input_shape=(8, 8, 1)),
    #     layers.Flatten(),
    #     # Add another:
    #     layers.Dense(64, activation='relu'),
    #     # Add a softmax layer with 10 output units:
    #     layers.Dense(10, activation='softmax')])

    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    input_layer = (layers.Dense(64, activation='relu',input_shape =(8,8,1)))


    model.add(input_layer)
    model.add(layers.Flatten())
    # Add another:
    model.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # # Configure a model for mean-squared error regression.
    # model.compile(optimizer=tf.train.AdamOptimizer(0.01),
    #               loss='mse',       # mean squared error
    #               metrics=['mae'])  # mean absolute error
    #
    # # Configure a model for categorical classification.
    # model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
    #               loss=tf.keras.losses.categorical_crossentropy,
    #               metrics=[tf.keras.metrics.categorical_accuracy])

    hist = model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=numEpochs,
                     validation_data=(X_test, y_test))
    # scores = model.evaluate(X_test, y_test, verbose=0)


if __name__== "__main__":
  main()
