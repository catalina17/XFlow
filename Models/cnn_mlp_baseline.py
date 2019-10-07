from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, merge, Input
from keras.layers.normalization import BatchNormalization

from data_config import *


def get_model():
	# Dataset config
	config = data_constants[args.dataset.lower()]
	inputCNNshape = config['inputCNNshape']
	inputMLPshape = config['inputMLPshape']
	nb_classes = config['nb_classes']

    # Build the CNN
    inputCNN = Input(shape=inputCNNshape)
    inputNorm = BatchNormalization()(inputCNN)

    conv = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputNorm)
    conv = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D((2,2), strides=(2, 2))(conv)
    pool = Dropout(0.25)(pool)

    conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(pool)
    conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D((2,2), strides=(2, 2))(conv)
    pool = Dropout(0.25)(pool)

    reshape = Flatten()(pool)
    fcCNN = Dense(256, activation='relu')(reshape)

    # Build the MLP
    inputMLP = Input(shape=inputMLPshape)

    fcMLP = Dense(128, activation='relu')(inputMLP)
    fcMLP = BatchNormalization()(fcMLP)
    fcMLP = Dropout(0.5)(fcMLP)

    fcMLP = Dense(128, activation='relu')(fcMLP)

    # Merge the models
    merged = merge([fcCNN, fcMLP], mode='concat')
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)

    fc = Dense(512, activation='relu')(merged)
    fc = Dropout(0.5)(fc)

    out = Dense(nb_classes, activation='softmax')(fc)

    # Return the model object
    model = Model(input=[inputCNN, inputMLP], output=out)
    return model
