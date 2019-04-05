This is a supervised learning project which conducts 3 types of experiments.
    The Dataset is from UC Irvine Machine Learning Repository. This can be found at
        https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    with the data being found at
        https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/

    Experiment 1: Compares Sum Of Squares error function to Cross entropy error function using different types of hyper
     parameters using ReLU as the activation function for the hidden layers and a softmax function for the output layer.
     For the best model it outputs confusion matrix.

    Experiment 2: Compares tanh vs ReLU as the activation function for the hidden units.

    Experiment 3: using a cross-entropy error function and ReLU activation function for hidden units calculate loss and
    accuracy using a convolutional neural network.

This project uses Python 3.6.7 with the following libraries
    Keras-Applications	1.0.7	1.0.7
    matplotlib	3.0.3	3.0.3
    numpy	1.16.2	1.16.2
    pandas	0.24.2	0.24.2
    pip	10.0.1	19.0.3
    seaborn	0.9.0	0.9.0
    tensorflow	1.13.1	1.13.1
    scikit-learn	0.20.3	0.20.3 (this is just used to create validation data. This can be done manually)
    tensorflow-gpu	1.13.1	1.13.1 (for convolutional neural network)

If you do not have a GPU you may want to comment out any code related to convolutional networks

The full set of experiments took a total of to run using 431.165 seconds a i7-4790k and a GTX 980.

Instructions for Running
    Ensure all libraries are installed as listed above.

    Project Structure is as follows:
    NeuralNetworkClassifier
        Data
            test
                optdigits.tes
            train
                examples.csv
                labels.csv
                optdigits.tra
                optdigits-orig.names.txt
        main.py

    Run main.py