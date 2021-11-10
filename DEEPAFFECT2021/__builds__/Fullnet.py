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
from __builds__.vgg16_places_365 import VGG16_Places365
from keras.layers import concatenate
from keras.regularizers import l2

class FullNet:
	@staticmethod
	def build_scene_branch(inputs, finalAct="linear", chanDim = -1):
		# initialize the scene branch and provide input shape,

		VGG_plc = VGG16_Places365(weights='places', include_top=True, input_shape=(224, 224, 3))

		# for layer in VGG_plc.layers:
		# 	layer.trainable = False

		x = VGG_plc(inputs)

		model = Model(inputs, x)

		return model

		# x = VGG16_Places365(weights='places', include_top=True, input_shape=(224, 224, 3))(inputs)
		# # define a branch for the scene clasification - check these are doing what they say they are doing
		#
		# model = Model(inputs, x)
		#
		# return model

	@staticmethod
	def build_object_branch (inputs, finalAct="linear", chanDim = -1):

		base_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

		# for layer in base_model.layers:
		# 	layer.trainable = False

		x = base_model(inputs)

		model = Model(inputs, x)

		return model


		# x = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))(inputs)
		# # if not using top then define an output layer, if not just return the model
		#
		# model = Model(inputs, x)
		#
		# return model

	@staticmethod
	def build_valence_branch(combinedInput, finalAct="relu"):
		#x = Dense(1365, activation='relu')(combinedInput)
		# x = BatchNormalization()(x)
		#x = Dropout(0.5)(x)


		x = Dense(3000, activation='relu')(combinedInput)
		# x = BatchNormalization()(x)
		x = Dropout(0.5)(x)

		x = Dense(1000, activation='relu')(x)
		# x = BatchNormalization()(x)
		x = Dropout(0.5)(x)

		x = Dense(500, activation='relu')(x)
		# x = BatchNormalization()(x)
		x = Dropout(0.5)(x)

		x = Dense(1)(x)
		x = Activation(finalAct, name="valence_output")(x)

		return x

	@staticmethod
	def build_arousal_branch(combinedInput, finalAct="relu"):
		#x = Dense(1365, activation='relu')(combinedInput)
		# x = BatchNormalization()(x)
		#x = Dropout(0.5)(x)

		#kernel_regularizer = l2(0.00001)

		x = Dense(3000, activation='relu')(combinedInput)
		# x = BatchNormalization()(x)
		x = Dropout(0.5)(x)

		x = Dense(1000, activation='relu')(x)
		# x = BatchNormalization()(x)
		x = Dropout(0.5)(x)

		x = Dense(500, activation='relu')(x)
		# x = BatchNormalization()(x)
		x = Dropout(0.5)(x)

		x = Dense(1)(x)
		x = Activation(finalAct, name="arousal_output")(x)

		return x

	@staticmethod
	def build(width, height, depth, finalAct="relu"):
		#receive the inputs from the other script (combined-MULT.py)

		#finalAct="relu"
		inputShape = (height, width, depth)
		chanDim = -1

		inputs = Input(shape=inputShape)

		#build both the scene and object categorisers
		sceneBranch = FullNet.build_scene_branch(inputs, finalAct=finalAct, chanDim=chanDim)
		objectBranch = FullNet.build_object_branch(inputs, finalAct=finalAct, chanDim=chanDim)

		#combined the out puts of each branch of the network

		combinedInput = concatenate([sceneBranch.output, objectBranch.output])

		valenceBranch = FullNet.build_valence_branch(combinedInput, finalAct=finalAct)
		arousalBranch = FullNet.build_arousal_branch(combinedInput, finalAct=finalAct)


		model = Model(inputs=inputs,
					  outputs=[valenceBranch, arousalBranch],
					  name="fullnet")

		return model


