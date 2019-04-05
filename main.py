import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn.model_selection as sk
from sklearn.metrics import confusion_matrix
import os

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)
tf.random.set_random_seed(88)


def main():
    # data is 64 + number which represents the given data's number so its 8 x 8
    # input matrix of 8x8 where each element is an integer in the range 0..16  + last value which is the number

    testing_df = pd.read_csv('data/test/optdigits.tes', header=None)
    X_testing, y_testing = testing_df.loc[:, 0:63], testing_df.loc[:, 64]

    training_df = pd.read_csv('data/train/optdigits.tra', header=None)
    X_training, y_training = training_df.loc[:, 0:63], training_df.loc[:, 64]

    X_training, X_validation, y_training, y_val = sk.train_test_split(X_training,  # validation data
                                                                      y_training,
                                                                      test_size=0.20,
                                                                      random_state=42)

    X_train = X_training.to_numpy().reshape(-1, 8, 8, 1)
    X_test = X_testing.to_numpy().reshape(-1, 8, 8, 1)
    X_validation = X_validation.to_numpy().reshape(-1, 8, 8, 1)

    y_validation = keras.utils.to_categorical(y_val, 10)
    y_train = keras.utils.to_categorical(y_training, 10)
    y_test = keras.utils.to_categorical(y_testing, 10)

    # print(y_validation.shape)
    # print(X_validation.shape)
    #
    # print(X_train.shape)
    # print(y_train.shape)
    #
    # print(X_test.shape)
    # print(y_test.shape)

    loss = 'categorical_crossentropy'
    x_evalData = X_test
    y_evalData = y_test
    # trainModels(1, 25, 64, 64, 0, .001, 0.0, X_train, y_train, X_validation, y_evalData, loss, 'relu',y_val)
    # trainModels(1, 25, 128, 64, 0.01, .05, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
    #             'relu',y_val)
    trainModels(5, 50, 64, 128, 0.001, .02, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', y_testing)
    # trainModels(5, 50, 128, 128, 0.1, .005, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
    #             'relu',y_val)
    # trainModels(5, 75, 10, 64, 0.001, .005, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
    #             'relu',y_val)
    # trainModels(20, 100, 120, 256, 0.01, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
    #             'relu',y_val)
    # trainModels(20, 100, 256, 64, 0.001, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
    #             'relu',y_val)

    # loss = 'mean_squared_error'
    # trainModels(1, 25, 256, 64, 0, .001, 0.0, X_train, y_train, X_test, y_test,X_validation, y_validation,loss,'relu')
    #
    # loss = 'categorical_crossentropy'
    # trainModels(1, 25, 256, 64, 0, .001, 0.0, X_train, y_train, X_test, y_test,X_validation, y_validation,loss,'relu')
    #
    # trainModels(1, 25, 256, 64, 0, .001, 0.0, X_train, y_train, X_test, y_test,X_validation, y_validation,loss,'tanh')


def trainModels(numHiddenLayers, numEpochs, numHiddenUnitsPerLayer, batchSize, momentum, learningRate, decay, X_train,
                y_train, x_evalData, y_evalData, lossFunction, hiddenUnitActivation, y_val):
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64, activation='relu', input_shape=(8, 8, 1)))

    model.add(layers.Flatten())

    for x in range(numHiddenLayers):
        # Add another:
        model.add(layers.Dense(numHiddenUnitsPerLayer, activation=hiddenUnitActivation))

    # output with softmax activation function
    model.add(layers.Dense(10, activation='softmax'))

    sgd = tf.keras.optimizers.SGD(
        lr=learningRate, momentum=momentum, decay=decay)

    model.compile(optimizer=sgd,
                  loss=lossFunction,
                  metrics=['accuracy'])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
    # ,tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    hist = model.fit(X_train, y_train, batch_size=batchSize, epochs=numEpochs, verbose=0)
    scores = model.evaluate(x_evalData, y_evalData, verbose=1)

    print("hidden layers, = ", numHiddenLayers, " hidden units = ", numHiddenUnitsPerLayer,
          " epochs = ", numEpochs, "Batch Size = ", batchSize, " learning rate = ", learningRate, " momentum rate = ",
          momentum, "Loss = "
          , scores[0], "Accuracy = ", scores[1])

    labels = (tf.argmax(y_val, axis=0))

    prediction = model.predict(x_evalData)
    confusion = confusion_matrix(y_val, np.argmax(prediction, axis=1))
    print(confusion)
    print(tf.keras.metrics.CategoricalAccuracy(model))


def printConfusionMatrix(model):
    numCorrect = np.zeros(21, 21)


if __name__ == "__main__":
    main()
