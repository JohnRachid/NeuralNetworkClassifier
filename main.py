import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn.model_selection as sk


def main():
    # data is 64 + number which represents the given data's number so its 8 x 8
    # input matrix of 8x8 where each element is an integer in the range 0..16  + last value which is the number

    testing_df = pd.read_csv('data/test/optdigits.tes', header=None)
    X_testing, y_testing = testing_df.loc[:, 0:63], testing_df.loc[:, 64]

    training_df = pd.read_csv('data/train/optdigits.tra', header=None)
    X_training, y_training = training_df.loc[:, 0:63], training_df.loc[:, 64]

    X_training, X_validation, y_training, y_validation = sk.train_test_split(X_training, #validation data
                                                        y_training,
                                                        test_size=0.20,
                                                        random_state=42)

    X_train = X_training.to_numpy().reshape(-1, 8, 8, 1)
    X_test = X_testing.to_numpy().reshape(-1, 8, 8, 1)
    X_validation = X_validation.to_numpy().reshape(-1, 8, 8, 1)

    y_validation = keras.utils.to_categorical(y_validation, 10)
    y_train = keras.utils.to_categorical(y_training, 10)
    y_test = keras.utils.to_categorical(y_testing, 10)

    print(y_validation.shape)
    print(X_validation.shape)

    print(X_train.shape)
    print(y_train.shape)

    print(X_test.shape)
    print(y_test.shape)

    trainModels(1, 25, 64, 64, 0, .001, 0.0, X_train, y_train, X_test, y_test,X_validation, y_validation)
    trainModels(1, 25, 64, 64, 0.001, .01, 0.0, X_train, y_train, X_test, y_test,X_validation, y_validation)
    trainModels(5, 50, 64, 64, 0.001, .01, 0.0, X_train, y_train, X_test, y_test,X_validation, y_validation)
    trainModels(5, 50, 64, 64, 0.1, .01, 0.0, X_train, y_train, X_test, y_test,X_validation, y_validation)
    trainModels(5, 50, 64, 64, 0.001, .1, 0.0, X_train, y_train, X_test, y_test,X_validation, y_validation)
    trainModels(20, 100, 64, 64, 0.001, .01, 0.0, X_train, y_train, X_test, y_test,X_validation, y_validation)
    trainModels(20, 100, 128, 64, 0.001, .01, 0.0, X_train, y_train, X_test, y_test,X_validation, y_validation)


def trainModels(numHiddenLayers, numEpochs, numHiddenUnitsPerLayer, batchSize, momentum, learningRate, decay, X_train, y_train, X_test, y_test , x_validation, y_validation):
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
    model.add(layers.Dense(64, activation='relu',input_shape =(8,8,1)))
    model.add(layers.Flatten())

    for x in range(numHiddenLayers):
        # Add another:
        model.add(layers.Dense(numHiddenUnitsPerLayer, activation='relu'))

    # output with softmax activation function
    model.add(layers.Dense(10, activation='softmax'))

    sgd = tf.keras.optimizers.SGD(
        lr=learningRate,momentum=momentum, decay=decay)

    model.compile(optimizer=sgd,
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
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,)]
            #,tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    hist = model.fit(X_train, y_train, batch_size=batchSize, epochs=numEpochs,verbose=0)
    scores = model.evaluate(x_validation, y_validation, verbose=1)
    print(scores)

    print("end of test with ", numHiddenLayers, "hidden layers, ", numHiddenUnitsPerLayer, " hidden units,",
          numEpochs, " epochs", learningRate, "learning rate", momentum, " momentum rate", scores[0],
          "loss ", scores[1], "accuracy")


if __name__== "__main__":
  main()
