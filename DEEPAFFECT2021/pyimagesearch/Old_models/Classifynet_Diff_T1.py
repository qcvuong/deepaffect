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
	def build_scene_branch(inputs, finalAct="linear", chanDim = -1):
		# initialize the input shape and channel dimension, assuming
		# TensorFlow/channels-last ordering

		x = VGG16_Places365(weights='places', include_top=False, input_shape=(64, 64, 3))(inputs)
		# define a branch of output layers for the number of different
		# clothing categories (i.e., shirts, jeans, dresses, etc.)
		x = Flatten()(x)
		x = Dense(256)(x)
		x = Activation("relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		x = Dense(1)(x)
		x = Activation(finalAct, name="scene_output")(x)

	# return the CNN
		return x

	@staticmethod
	def build_object_branch (inputs, finalAct="linear", chanDim = -1):

		x = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))(inputs)
		# define a branch of output layers for the number of different
		# colors (i.e., red, black, blue, etc.)
		x = Flatten()(x)
		x = Dense(128)(x)
		x = Activation("relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		x = Dense(1)(x)
		x = Activation(finalAct, name="object_output")(x)

		# return the color prediction sub-network
		return x

	@staticmethod
	def build(width, height, depth, finalAct="linear"):

		#finalAct="relu"
		inputShape = (height, width, depth)
		chanDim = -1

		inputs = Input(shape=inputShape)

		sceneBranch = ClassifyNet.build_scene_branch(inputs, finalAct=finalAct, chanDim=chanDim)
		objectBranch = ClassifyNet.build_object_branch(inputs, finalAct=finalAct, chanDim=chanDim)

		model = Model(inputs=inputs,
					  outputs=[sceneBranch, objectBranch],
					  name="classifynet")

		return model

