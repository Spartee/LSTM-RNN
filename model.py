import os
import time
import warnings
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


def build_LSTM_model(layers, optimize, deep=False):
    """Constructs the LSTM model using the layers from Keras. Dropout layers
       added at the end of each layer to help prevent overfitting. Linear
       activation function for output layer. Optimizer can vary as a parameter
       but rmsprop usually does the best as it is meant for RNN's
       TODO: Assert layer size for all models"""

    LSTM_model = Sequential()

    # Input layer
    LSTM_model.add(LSTM(input_shape=(layers[1], layers[0]), output_dim=layers[1], return_sequences=True))
    LSTM_model.add(Dropout(.2))

    if deep:

        LSTM_model.add(LSTM(input_shape=(layers[1], layers[0]), output_dim=layers[2], return_sequences=True))
        LSTM_model.add(Dropout(.2))

        # Hidden Layer
        LSTM_model.add(LSTM(layers[3], return_sequences=False))
        LSTM_model.add(Dropout(.2))

        # Output layer
        LSTM_model.add(Dense(output_dim=layers[4]))
        LSTM_model.add(Activation("linear"))

    else:

        # Hidden Layer
        LSTM_model.add(LSTM(layers[2], return_sequences=False))
        LSTM_model.add(Dropout(.2))

        # Output layer
        LSTM_model.add(Dense(output_dim=layers[3]))
        LSTM_model.add(Activation("linear"))

    # time the compilation
    start = time.time()
    LSTM_model.compile(loss="mse", optimizer=optimize)
    print("LSTM model took ", time.time() - start, " time to compile")

    return LSTM_model
