from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers import merge, Input
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization

from data_config import *


def get_model():
	# Dataset config
	assert args.dataset.lower() == 'avletters'
	config = data_constants['avletters']
	inputCNNshape = config['inputCNNshape']
	inputMLPshape = config['inputMLPshape']
	nb_classes = config['nb_classes']

	# Build the CNN - pre-cross-connections
	inputCNN = Input(shape=inputCNNshape)
	inputNorm = BatchNormalization()(inputCNN)

	conv = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(inputNorm)
	conv = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(conv)
	conv = BatchNormalization()(conv)

	pool = MaxPooling2D((2,2), strides=(2, 2))(conv)
	pool = Dropout(0.25)(pool)

	# Build the MLP - pre-cross-connections
	inputMLP = Input(shape=inputMLPshape)

	fcMLP = Dense(128, activation='relu')(inputMLP)
	fcMLP = BatchNormalization()(fcMLP)
	fcMLP = Dropout(0.5)(fcMLP)


	# Add the 1st round of cross-connections - CNN to MLP
	x21 = Convolution2D(16, 1, 1, border_mode='same')(pool)
	x21 = PReLU()(x21)
	x21 = BatchNormalization()(x21)
	x21 = Flatten()(x21)
	x21 = Dense(64)(x21)
	x21 = PReLU()(x21)

	# Add 1st shortcut (residual connection) from CNN input to MLP
	short1_2dto1d = MaxPooling2D((4,4), strides=(4,4))(inputNorm)
	short1_2dto1d = Dropout(0.25)(short1_2dto1d)
	short1_2dto1d = Flatten()(short1_2dto1d)
	short1_2dto1d = Dense(128)(short1_2dto1d)
	short1_2dto1d = PReLU()(short1_2dto1d)

	# MLP to CNN - cross-connections
	x12 = Dense(33*23)(fcMLP)
	x12 = PReLU()(x12)
	x12 = BatchNormalization()(x12)
	x12 = Dropout(0.5)(x12)
	x12 = Reshape((33,23,1))(x12)
	x12 = Deconvolution2D(16, 8, 8, output_shape=(None, 40, 30, 16), border_mode='valid')(x12)
	x12 = PReLU()(x12)

	# 1st shortcut (residual connection) from MLP input to CNN
	short1_1dto2d = Dense(33*23)(inputMLP)
	short1_1dto2d = PReLU()(short1_1dto2d)
	short1_1dto2d = BatchNormalization()(short1_1dto2d)
	short1_1dto2d = Dropout(0.25)(short1_1dto2d)
	short1_1dto2d = Reshape((33,23,1))(short1_1dto2d)
	short1_1dto2d = Deconvolution2D(16, 8, 8, output_shape=(None, 40, 30, 16), border_mode='valid')(
		short1_1dto2d)
	short1_1dto2d = PReLU()(short1_1dto2d)
	short1_1dto2d = BatchNormalization()(short1_1dto2d)


	# CNN - post-cross-connections 1
	pool = merge([pool, short1_1dto2d], mode='sum')
	x12map = merge([x12, pool], mode='concat', concat_axis=3)
	x12map = BatchNormalization()(x12map)
	x12map = Dropout(0.5)(x12map)

	conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(x12map)
	conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(conv)
	conv = BatchNormalization()(conv)

	pool = MaxPooling2D((2,2), strides=(2, 2))(conv)
	pool = Dropout(0.25)(pool)

	# MLP - post-cross-connections 1
	fcMLP = merge([fcMLP, short1_2dto1d], mode='sum')
	x21map = merge([x21, fcMLP], mode='concat', concat_axis=1)
	x21map = BatchNormalization()(x21map)
	x21map = Dropout(0.5)(x21map)

	fcMLP = Dense(128, activation='relu')(x21map)
	fcMLP = BatchNormalization()(fcMLP)
	fcMLP = Dropout(0.5)(fcMLP)


	# Add the 2nd round of cross-connections - CNN to MLP
	x21 = Convolution2D(32, 1, 1, border_mode='same')(pool)
	x21 = PReLU()(x21)
	x21 = BatchNormalization()(x21)
	x21 = Flatten()(x21)
	x21 = Dense(128)(x21)
	x21 = PReLU()(x21)

	# Add 2nd shortcut (residual connection) from CNN input to MLP
	short2_2dto1d = MaxPooling2D((8,8), strides=(8,4))(inputNorm)
	short2_2dto1d = Dropout(0.25)(short2_2dto1d)
	short2_2dto1d = Flatten()(short2_2dto1d)
	short2_2dto1d = Dense(128)(short2_2dto1d)
	short2_2dto1d = PReLU()(short2_2dto1d)

	# MLP to CNN - cross-connections
	x12 = Dense(17*12)(fcMLP)
	x12 = PReLU()(x12)
	x12 = BatchNormalization()(x12)
	x12 = Dropout(0.5)(x12)
	x12 = Reshape((17,12,1))(x12)
	x12 = Deconvolution2D(32, 4, 4, output_shape=(None, 20, 15, 32), border_mode='valid')(x12)
	x12 = PReLU()(x12)

	# 2nd shortcut (residual connection) from MLP input to CNN
	short2_1dto2d = Dense(17*12)(inputMLP)
	short2_1dto2d = PReLU()(short2_1dto2d)
	short2_1dto2d = BatchNormalization()(short2_1dto2d)
	short2_1dto2d = Dropout(0.5)(short2_1dto2d)
	short2_1dto2d = Reshape((17,12,1))(short2_1dto2d)
	short2_1dto2d = Deconvolution2D(32, 4, 4, output_shape=(None, 20, 15, 32), border_mode='valid')(
		short2_1dto2d)
	short2_1dto2d = PReLU()(short2_1dto2d)


	# CNN - post cross-connections 2
	pool = merge([pool, short2_1dto2d], mode='sum')
	x12map = merge([x12, pool], mode='concat', concat_axis=3)
	x12map = BatchNormalization()(x12map)
	x12map = Dropout(0.5)(x12map)

	reshape = Flatten()(x12map)
	fcCNN = Dense(256, activation='relu')(reshape)


	# Merge the CNN and MLP models
	fcMLP = merge([fcMLP, short2_2dto1d], mode='sum')
	merged = merge([x21, fcCNN, fcMLP], mode='concat')
	merged = BatchNormalization()(merged)
	merged = Dropout(0.5)(merged)

	fc = Dense(512, activation='relu')(merged)
	fc = Dropout(0.5)(fc)

	out = Dense(nb_classes, activation='softmax')(fc)

	# Return the model object
	model = Model(input=[inputCNN, inputMLP], output=out)
	return model
