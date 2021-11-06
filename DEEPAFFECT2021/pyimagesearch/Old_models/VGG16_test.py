# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras import backend as K
from pyimagesearch.vgg16_places_365 import VGG16_Places365



def create_scene (width, height, depth, regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	#model = Sequential()
	inputShape = (height, width, depth)
	chanDim = -1

	# if we are using "channels first", update the input shape
	# and channels dimension
	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)
		chanDim = 1

	conv_base = VGG16_Places365(weights='places',
					  include_top=False,
					  input_shape=inputShape)

	model = Sequential()
	model.add(conv_base)
	model.add(Flatten())
	model.add(Dense(16))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Dropout(0.5))

	model.add(Dense(4))
	model.add(Activation("relu"))

	if regress:
		model.add(Dense(1, activation="linear"))

	return model

def create_VGG16 (width, height, depth, regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	#model = Sequential()
	inputShape = (height, width, depth)
	chanDim = -1

	# if we are using "channels first", update the input shape
	# and channels dimension
	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)
		chanDim = 1

	conv_base = VGG16(weights='imagenet',
					  include_top=False,
					  input_shape=inputShape)

	model = Sequential()
	model.add(conv_base)
	model.add(Flatten())
	model.add(Dense(16))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Dropout(0.5))

	model.add(Dense(4))
	model.add(Activation("relu"))

	if regress:
		model.add(Dense(1, activation="linear"))

	return model

def create_cnn(width, height, depth, regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	model = Sequential()
	inputShape = (height, width, depth)
	chanDim = -1

	# if we are using "channels first", update the input shape
	# and channels dimension
	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)
		chanDim = 1

	# CONV => RELU => POOL
	model.add(Conv2D(32, (3, 3), padding="same",
					 input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	model.add(Dropout(0.25))

	# (CONV => RELU) * 2 => POOL
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# (CONV => RELU) * 2 => POOL
	model.add(Conv2D(128, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(128, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	if regress:
		model.add(Dense(1, activation="linear"))


	# softmax classifier
	#model.add(Dense(classes))
	#model.add(Activation(finalAct))

	# return the constructed network architecture
	return model