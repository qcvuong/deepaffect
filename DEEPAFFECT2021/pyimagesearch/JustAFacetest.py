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
from keras.layers import concatenate

class ClassifyNetTest:
	@staticmethod
	def build_face_branch(inputs, chanDim = -1):
	# CONV => RELU => POOL
		x = Conv2D(32, (3, 3), padding="same", input_shape=(224, 224, 3))(inputs)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(3, 3))(x)
		x = Dropout(0.25)(x)

		# (CONV => RELU) * 2 => POOL
		x = Conv2D(64, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Conv2D(64, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)

	# (CONV => RELU) * 2 => POOL
		x = Conv2D(128, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Conv2D(128, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)

	# first (and only) set of FC => RELU layers
		x = Flatten()(x)
		x = Dense(1024)(x)
		x = Activation("relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)

	# softmax classifier
		x = Dense(7)(x)
		x = Activation("softmax")(x)

		model = Model(inputs, x)

		model.load_weights("best_224_weights.h5")

		return model
	# return the constructed network architecture

	@staticmethod
	def build_scene_branch(inputs, finalAct="linear", chanDim = -1):
		# initialize the scene branch and provide input shape, assuming
		# TensorFlow/channels-last ordering

		x = VGG16_Places365(weights='places', include_top=True, input_shape=(224, 224, 3))(inputs)
		# define a branch for the scene clasification - check these are doing what they say they are doing

		#using VGG16_places with 365 scene categories
		#if not using top then can use the bellow fully connected head
		#x = Flatten()(x)
		#x = Dense(256)(x)
		#x = Activation("relu")(x)
		#x = BatchNormalization()(x)
		#x = Dropout(0.5)(x)
		#x = Dense(1)(x)
		#x = Activation(finalAct, name="scene_output")(x)

		model = Model(inputs, x)
	# return the CNN
		#return x

		return model

	@staticmethod
	def build_object_branch (inputs, finalAct="linear", chanDim = -1):

		x = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))(inputs)
		# if not using top then define an output layer, if not just return the model

		#x = Flatten()(x)
		#x = Dense(128)(x)
		#x = Activation("relu")(x)
		#x = BatchNormalization()(x)
		#x = Dropout(0.5)(x)
		#x = Dense(1)(x)
		#x = Activation(finalAct, name="object_output")(x)

		model = Model(inputs, x)
		# return the object prediction sub-network
		#return x

		return model


	@staticmethod
	def build(width, height, depth, finalAct="linear"):
	#receive the inputs from the other script (combined-MULT.py)

		#finalAct="relu"
		inputShape = (height, width, depth)
		chanDim = -1

		inputs = Input(shape=inputShape)

		#build both the scene and object categorisers
		sceneBranch = ClassifyNetTest.build_scene_branch(inputs, finalAct=finalAct, chanDim=chanDim)
		objectBranch = ClassifyNetTest.build_object_branch(inputs, finalAct=finalAct, chanDim=chanDim)
		faceBranch = ClassifyNetTest.build_face_branch(inputs, chanDim=chanDim)
		#combined the out puts of each branch of the network

		combinedInput = concatenate([sceneBranch.output, objectBranch.output, faceBranch.output])

		#use the combined output as the input for a fully connected head
		#leading to a single contineous value as the output


		x = Dense(3000, activation='relu')(combinedInput)
		x = Dropout(0.5)(x)

		x = Dense(1000, activation='relu')(x)
		x = Dropout(0.5)(x)

		x = Dense(1000, activation='relu')(x)
		x = Dropout(0.5)(x)

		x = Dense(1, activation=finalAct)(x)

		model = Model(inputs=inputs,
					  outputs=x,
					  name="classifynet")

		return model


