from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, merge, Input
from keras.layers import LSTM, TimeDistributed, Masking, Reshape
from keras.layers.normalization import BatchNormalization

from Datasets.data_config import *


def get_model(args):
    # Dataset config
    config = data_constants[args.dataset.lower()]
    inputCNNshape = config['lstm_inputCNNshape']
    inputMLPshape = config['lstm_inputMLPshape']
    nb_classes = config['nb_classes']

    # Build the CNN
    inputCNN = Input(shape=inputCNNshape)
    inputNorm = TimeDistributed(Flatten())(inputCNN)
    inputNorm = Masking(mask_value=0.)(inputNorm)
    inputNorm = TimeDistributed(Reshape((90, 120, 1)))(inputNorm)
    inputNorm = BatchNormalization(axis=1)(inputNorm)

    conv = TimeDistributed(
        Convolution2D(8, 3, 3, border_mode='same', activation='relu'), name='conv11')(inputNorm)
    pool = TimeDistributed(MaxPooling2D((2,2), strides=(2, 2)), name='maxpool1')(conv)

    conv = TimeDistributed(
        Convolution2D(16, 3, 3, border_mode='same', activation='relu'), name='conv21')(pool)
    pool = TimeDistributed(MaxPooling2D((2,2), strides=(2, 2)), name='maxpool2')(conv)

    reshape = TimeDistributed(Flatten(), name='flatten1')(pool)
    fcCNN = TimeDistributed(Dense(64, activation='relu'), name='fcCNN')(reshape)

    # Build the MLP
    inputMLP = Input(shape=inputMLPshape)
    inputMasked = Masking(mask_value=0., input_shape=inputMLPshape)(inputMLP)

    fcMLP = TimeDistributed(Dense(32, activation='relu'), name='fc1')(inputMasked)

    fcMLP = TimeDistributed(Dense(32, activation='relu'), name='fc2')(fcMLP)

    # Merge the models
    merged = merge([fcCNN, fcMLP], mode='concat')
    merged = BatchNormalization(axis=1, name='mergebn')(merged)
    merged = Dropout(0.25, name='mergedrop')(merged)

    lstm = LSTM(64)(merged)
    out = Dense(nb_classes, activation='softmax')(lstm)

    # Return the model object
    model = Model(input=[inputCNN, inputMLP], output=out)
    return model
