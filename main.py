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
import time

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)
tf.random.set_random_seed(88)
#try to cut down on randomness
start = time.perf_counter()


def main():
    # data comes in as 65 chars which is 64 for the image pixels and 1 for the label
    # turn this into a 8x8x1

    testing_df = pd.read_csv('data/test/optdigits.tes', header=None)
    X_testing, y_testing = testing_df.loc[:, 0:63], testing_df.loc[:, 64]

    training_df = pd.read_csv('data/train/optdigits.tra', header=None)
    X_training, y_training = training_df.loc[:, 0:63], training_df.loc[:, 64]

    X_training, X_validation, y_training, y_val = sk.train_test_split(X_training,  # validation data
                                                                      y_training,
                                                                      test_size=0.20,
                                                                      random_state=42)#this random shuffle witll cause inconsistency in data
    # shapping data so it can be used for normal NN and conv NN
    X_train = X_training.to_numpy().reshape(-1, 8, 8, 1)
    X_test = X_testing.to_numpy().reshape(-1, 8, 8, 1)
    X_validation = X_validation.to_numpy().reshape(-1, 8, 8, 1)

    # this one hot encodes the data. You shouldn't use this for the confusion matrices
    y_validation = keras.utils.to_categorical(y_val, 10)
    y_train = keras.utils.to_categorical(y_training, 10)
    y_test = keras.utils.to_categorical(y_testing, 10)

    x_evalData = X_test  # X_validation for 20% of training data used as validation data, X_test for testing data
    y_evalData = y_test  # y_validation for 20% of training data used as validation data, y_test for testing data
    labels = y_testing  # used for confusion matrix. use y_val when using X_validation and y_validation otherwise use y_testing

    loss = 'categorical_crossentropy'
    # categorical_crossentropy experiments
    trainModels(1, 25, 64, 64, 0, .001, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels)
    trainModels(1, 25, 128, 64, 0.01, .05, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', labels)
    trainModels(5, 50, 64, 128, 0.001, .02, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', labels)

    trainModels(5, 50, 128, 128, 0.1, .005, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', labels)
    trainModels(5, 75, 10, 64, 0.001, .005, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', labels)
    trainModels(20, 100, 120, 256, 0.01, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', labels)
    trainModels(20, 100, 256, 64, 0.001, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', labels)

    loss = 'mean_squared_error'
    # mean_squared_error experiments
    trainModels(1, 25, 64, 64, .006, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels)
    trainModels(1, 25, 64, 64, 0, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels)
    trainModels(1, 25, 128, 64, 0.01, .05, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', labels)
    trainModels(5, 50, 64, 128, 0.001, .02, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', labels)

    trainModels(5, 50, 128, 128, 0.1, .05, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', labels)
    trainModels(10, 100, 120, 256, 0.01, .5, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', labels)
    trainModels(10, 100, 256, 64, 0.01, .09, 0.0, X_train, y_train, x_evalData, y_evalData, loss,
                'relu', labels)

    # categorical crossentropy tanh experiments
    # loss = 'categorical_crossentropy'
    trainModels(1, 25, 100, 64, .02, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'tanh', labels)
    trainModels(2, 30, 50, 32, 0.002, .003, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'tanh', labels)
    trainModels(3, 60, 90, 64, 0.03, .02, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'tanh', labels)
    trainModels(5, 50, 50, 128, 0.06, .001, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'tanh', labels)
    trainModels(10, 75, 30, 80, 0.05, .02, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'tanh', labels)
    trainModels(20, 90, 30, 256, 0.2, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'tanh', labels)
    trainModels(30, 100, 30, 64, 0.08, .02, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'tanh', labels)
    loss = 'categorical_crossentropy'
    # categorical crossentropy relu experiments2
    trainModels(1, 25, 100, 64, .02, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels)
    trainModels(2, 30, 50, 32, 0.002, .003, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels)
    trainModels(3, 60, 90, 64, 0.01, .03, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels)

    trainModels(5, 50, 50, 128, 0.06, .001, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels)
    trainModels(10, 75, 30, 80, 0.05, .02, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels)
    trainModels(20, 90, 30, 256, 0.02, .001, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels)
    trainModels(30, 100, 30, 64, 0.02, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels)

    loss = 'categorical_crossentropy'
    # Convolutional experiments relu experiments2
    trainConvModels(1, 30, 10, 128, 0.02, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels, 10,
                    400, 5, .3)
    trainConvModels(2, 30, 15, 128, 0.02, .05, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels, 5,
                    200, 3, .3)
    trainConvModels(3, 90, 20, 128, 0.02, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels, 10,
                    400, 5, .2)

    trainConvModels(4, 120, 25, 128, 0.02, .02, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels, 10,
                    200, 6, .6)
    trainConvModels(5, 150, 30, 128, 0.02, .01, 0.0, X_train, y_train, x_evalData, y_evalData, loss, 'relu', labels, 10,
                    400, 5, .5)

    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)


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
    # This employs the early stopping technique. Since the patience is 2 the model will stop training if there is
    # no improvement in 2 epochs
    # ,tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)] #saving checkpoint

    hist = model.fit(X_train, y_train, batch_size=batchSize, epochs=numEpochs, verbose=0)
    scores = model.evaluate(x_evalData, y_evalData, verbose=1)

    print("hidden layers, = ", numHiddenLayers, " hidden units = ", numHiddenUnitsPerLayer,
          " epochs = ", numEpochs, "Batch Size = ", batchSize, " learning rate = ", learningRate, " momentum rate = ",
          momentum, "Loss = "
          , scores[0], "Accuracy = ", scores[1])

    labels = (tf.argmax(y_val, axis=0))

    prediction = model.predict(x_evalData)
    printConfusionMatrix(y_val, prediction)


def trainConvModels(numHiddenLayers, numEpochs, numHiddenUnitsPerLayer, batchSize, momentum, learningRate, decay,
                    X_train, y_train, x_evalData, y_evalData, lossFunction, hiddenUnitActivation, y_val,
                    kernalSize, filters, poolSize, dropout):
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(tf.keras.layers.Convolution2D(filters, kernalSize, input_shape=(8, 8, 1), data_format="channels_first",
                                            activation='relu',
                                            padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(poolSize, poolSize),
                                           data_format="channels_last"))  # channel dimension/depth is dim 1 to match 32 x 3 x 3
    model.add(layers.Dropout(dropout))
    model.add(tf.keras.layers.Flatten())

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

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0)]
    # This employs the early stopping technique. Since the patience is 2 the model will stop training if there is
    # no improvement in 2 epochs
    # ,tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)] #saving checkpoint

    hist = model.fit(X_train, y_train, batch_size=batchSize, epochs=numEpochs, verbose=0)
    scores = model.evaluate(x_evalData, y_evalData, verbose=1)

    print("hidden layers, = ", numHiddenLayers, " hidden units = ", numHiddenUnitsPerLayer,
          " epochs = ", numEpochs, "Batch Size = ", batchSize, " learning rate = ", learningRate, " momentum rate = ",
          momentum, "Loss = "
          , scores[0], "Accuracy = ", scores[1])

    # labels = (tf.argmax(y_val, axis=0))

    prediction = model.predict(x_evalData)
    printConfusionMatrix(y_val, prediction)


def printConfusionMatrix(y_val, prediction):
    confusion = confusion_matrix(y_val, np.argmax(prediction, axis=1))
    print(confusion)


if __name__ == "__main__":
    main()
