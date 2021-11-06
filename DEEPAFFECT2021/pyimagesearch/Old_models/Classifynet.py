# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras import backend as K
from pyimagesearch.vgg16_places_365 import VGG16_Places365

class ClassifyNet:
	@staticmethod
	def build_scene_branch(width, height, depth, finalAct="relu", chanDim = -1):
		# initialize the input shape and channel dimension, assuming
		# TensorFlow/channels-last ordering
		inputShape = (height, width, depth)
		#chanDim = -1
		inputs = Input(shape=inputShape)

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		conv_base = VGG16_Places365(weights='places',
					  include_top=False,
					  input_shape=inputShape)

		x = conv_base(inputs)
		# flatten the volume, then FC => RELU => BN => DROPOUT
		x = Flatten()(x)
		x = Dense(16)(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Dropout(0.5)(x)

		# apply another FC layer, this one to match the number of nodes
		# coming out of the MLP
		x = Dense(1)(x)
		x = Activation(finalAct, name="scene_output")(x)

		# check to see if the regression node should be added
		#if regress:
		#x = Dense(1, activation="linear")(x)

	# construct the CNN
	#model = Model(inputs, x)

	# return the CNN
		return x

	@staticmethod
	def build_object_branch (width, height, depth, finalAct="relu", chanDim = -1):
		# initialize the input shape and channel dimension, assuming
		# TensorFlow/channels-last ordering
		#model = Sequential()
		inputShape = (height, width, depth)
		#chanDim = -1
		inputs = Input(shape=inputShape)
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		conv_base = VGG16(weights='imagenet',
					  include_top=False,
					  input_shape=inputShape)


		x = conv_base(inputs)
		# flatten the volume, then FC => RELU => BN => DROPOUT
		x = Flatten()(x)
		x = Dense(16)(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Dropout(0.5)(x)

		# apply another FC layer, this one to match the number of nodes
		# coming out of the MLP
		x = Dense(1)(x)
		x = Activation(finalAct, name="object_output")(x)

		# check to see if the regression node should be added
		#if regress:
		#x = Dense(1, activation="linear")(x)

	# construct the CNN
		#model = Model(inputs, x)

	# return the CNN
		return x

	@staticmethod
	def build(width, height, depth, finalAct="relu"):

		#finalAct="relu"
		inputShape = (height, width, depth)
		chanDim = -1

		inputs = Input(shape=inputShape)

		sceneBranch = ClassifyNet.build_scene_branch(width, height, depth, finalAct=finalAct, chanDim=chanDim)
		objectBranch = ClassifyNet.build_object_branch(width, height, depth, finalAct=finalAct, chanDim=chanDim)

		model = Model(inputs=inputs,
					  outputs=[sceneBranch, objectBranch],
					  name="classifynet")

		return model


