from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers import add, concatenate, Input
from keras.layers import LSTM, TimeDistributed, Masking
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization

from Datasets.data_config import *


def get_model(args):
    # Dataset config
    assert args.dataset.lower() == 'avletters'
    config = data_constants['avletters']
    inputCNNshape = config['lstm_inputCNNshape']
    inputMLPshape = config['lstm_inputMLPshape']
    nb_classes = config['nb_classes']

    # Build the CNN - pre-cross-connections
    inputCNN = Input(shape=inputCNNshape)
    inputNorm = TimeDistributed(Flatten())(inputCNN)
    inputNorm = Masking(mask_value=0.)(inputNorm)
    inputNorm = TimeDistributed(Reshape((80, 60, 1)))(inputNorm)
    inputNorm = BatchNormalization(axis=1)(inputNorm)

    conv = TimeDistributed(
        Convolution2D(8, 3, 3, border_mode='same', activation='relu'), name='conv11')(inputNorm)
    pool = TimeDistributed(
        MaxPooling2D((2,2), strides=(2, 2)), name='maxpool1')(conv)

    # Build the MLP - pre-cross-connections
    inputMLP = Input(shape=inputMLPshape)
    inputMasked = Masking(mask_value=0., input_shape=inputMLPshape)(inputMLP)

    fcMLP = TimeDistributed(
        Dense(32, activation='relu'), name='fc1')(inputMasked)

    # Add the 1st round of cross-connections - CNN to MLP
    x21 = TimeDistributed(Convolution2D(8, 1, 1, border_mode='same'))(pool)
    x21 = TimeDistributed(PReLU())(x21)
    x21 = TimeDistributed(Flatten())(x21)
    x21 = TimeDistributed(Dense(32))(x21)
    x21 = TimeDistributed(PReLU())(x21)

    # Add 1st shortcut (residual connection) from CNN input to MLP
    short1_2dto1d = TimeDistributed(MaxPooling2D((4,4), strides=(4,4)))(inputNorm)
    short1_2dto1d = TimeDistributed(Flatten())(short1_2dto1d)
    short1_2dto1d = TimeDistributed(Dense(32))(short1_2dto1d)
    short1_2dto1d = TimeDistributed(PReLU())(short1_2dto1d)

    # Cross-connections - MLP to CNN
    x12 = TimeDistributed(Dense(25*15))(fcMLP)
    x12 = TimeDistributed(PReLU())(x12)
    x12 = TimeDistributed(Reshape((25,15,1)))(x12)
    x12 = TimeDistributed(Conv2DTranspose(8, (16, 16), padding='valid'))(x12)
    x12 = TimeDistributed(PReLU())(x12)

    # 1st shortcut (residual connection) from MLP input to CNN
    short1_1dto2d = TimeDistributed(Dense(25*15))(inputMasked)
    short1_1dto2d = TimeDistributed(PReLU())(short1_1dto2d)
    short1_1dto2d = TimeDistributed(Reshape((25,15,1)))(short1_1dto2d)
    short1_1dto2d = TimeDistributed(Conv2DTranspose(8, (16, 16), padding='valid'))(short1_1dto2d)
    short1_1dto2d = TimeDistributed(PReLU())(short1_1dto2d)


    # CNN - post-cross-connections 1
    pool = add([pool, short1_1dto2d])
    merged = concatenate([pool, x12])

    conv = TimeDistributed(
        Convolution2D(16, 3, 3, border_mode='same', activation='relu'), name='conv21')(merged)
    pool = TimeDistributed(
        MaxPooling2D((2,2), strides=(2, 2)), name='maxpool2')(conv)


    # MLP - post-cross-connections 1
    fcMLP = add([fcMLP, short1_2dto1d])
    fcMLP = concatenate([fcMLP, x21])

    fcMLP = TimeDistributed(
        Dense(32, activation='relu'), name='fc2')(fcMLP)

    # Add the 2nd round of cross-connections - CNN to MLP
    x21 = TimeDistributed(Convolution2D(16, 1, 1, border_mode='same'))(pool)
    x21 = TimeDistributed(PReLU())(x21)
    x21 = TimeDistributed(Flatten())(x21)
    x21 = TimeDistributed(Dense(64))(x21)
    x21 = TimeDistributed(PReLU())(x21)

    # Add 2nd shortcut (residual connection) from CNN input to MLP
    short2_2dto1d = TimeDistributed(MaxPooling2D((8,8), strides=(8,4)))(inputNorm)
    short2_2dto1d = TimeDistributed(Flatten())(short2_2dto1d)
    short2_2dto1d = TimeDistributed(Dense(32))(short2_2dto1d)
    short2_2dto1d = TimeDistributed(PReLU())(short2_2dto1d)

    # Cross-connections - MLP to CNN
    x12 = TimeDistributed(Dense(13*8))(fcMLP)
    x12 = TimeDistributed(PReLU())(x12)
    x12 = TimeDistributed(Reshape((13,8,1)))(x12)
    x12 = TimeDistributed(Conv2DTranspose(16, (8, 8), padding='valid'))(x12)
    x12 = TimeDistributed(PReLU())(x12)

    # 2nd shortcut (residual connection) from MLP input to CNN
    short2_1dto2d = TimeDistributed(Dense(13*8))(inputMasked)
    short2_1dto2d = TimeDistributed(PReLU())(short2_1dto2d)
    short2_1dto2d = TimeDistributed(Reshape((13,8,1)))(short2_1dto2d)
    short2_1dto2d = TimeDistributed(Conv2DTranspose(16, (8, 8), padding='valid'))(short2_1dto2d)
    short2_1dto2d = TimeDistributed(PReLU())(short2_1dto2d)


    # CNN - post-cross-connections 2
    pool = add([pool, short2_1dto2d])
    merged = concatenate([pool, x12])

    reshape = TimeDistributed(
        Flatten(), name='flatten1')(merged)
    fcCNN = TimeDistributed(
        Dense(64, activation='relu'), name='fcCNN')(reshape)


    # Merge the models
    fcMLP = add([fcMLP, short2_2dto1d])
    merged = concatenate([fcCNN, fcMLP, x21])
    merged = BatchNormalization(axis=1, name='mergebn')(merged)
    merged = Dropout(0.5, name='mergedrop')(merged)

    lstm = LSTM(64)(merged)
    out = Dense(nb_classes, activation='softmax')(lstm)

    # Return the model object
    model = Model(input=[inputCNN, inputMLP], output=out)
    return model
