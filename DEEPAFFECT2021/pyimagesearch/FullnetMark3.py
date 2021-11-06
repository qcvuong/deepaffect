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
from keras.applications.vgg19 import VGG19
from keras import backend as K
from pyimagesearch.vgg16_places_365 import VGG16_Places365
from keras.layers import concatenate

class FullNet3:
	@staticmethod
	def build_face_branch(inputs, chanDim=-1):
		x = VGG19(include_top=False, weights=None, input_shape=(224, 224, 3))(inputs)

		x = Flatten()(x)

		x = Dense(2048)(x)
		x = Activation("relu")(x)
		#x = BatchNormalization()(x)
		#x = Dropout(0.5)(x)

		x = Dense(1024)(x)
		x = Activation("relu")(x)
		#x = BatchNormalization()(x)
		#x = Dropout(0.5)(x)

		x = Dense(512)(x)
		x = Activation("relu")(x)
		#x = BatchNormalization()(x)
		#x = Dropout(0.5)(x)

		# softmax classifier
		x = Dense(9)(x)
		x = Activation("softmax")(x)

		model = Model(inputs, x)

		model.load_weights("aa9EmotionBest.h5")
## aa9EmotionBest.h5
##VGG19_3_EarlyStop.h5
		return model

	# return the constructed network architecture
	@staticmethod
	def build_scene_branch(inputs, finalAct="linear", chanDim = -1):
		# initialize the scene branch and provide input shape,

		x = VGG16_Places365(weights='places', include_top=True, input_shape=(224, 224, 3))(inputs)
		# define a branch for the scene clasification - check these are doing what they say they are doing



		model = Model(inputs, x)


		return model

	@staticmethod
	def build_object_branch (inputs, finalAct="linear", chanDim = -1):

		x = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))(inputs)
		# if not using top then define an output layer, if not just return the model

		model = Model(inputs, x)

		return model

	@staticmethod
	def build_valence_branch(combinedInput, finalAct="linear"):
		x = Dense(3000, activation='relu')(combinedInput)
		x = Dropout(0.5)(x)

		x = Dense(1000, activation='relu')(x)
		x = Dropout(0.5)(x)

		x = Dense(500, activation='relu')(x)
		x = Dropout(0.5)(x)

		x = Dense(1)(x)
		x = Activation('relu', name="valence_output")(x)

		return x

	@staticmethod
	def build_arousal_branch(combinedInput, finalAct="linear"):
		x = Dense(3000, activation='relu')(combinedInput)
		x = Dropout(0.5)(x)

		x = Dense(1000, activation='relu')(x)
		x = Dropout(0.5)(x)

		x = Dense(500, activation='relu')(x)
		x = Dropout(0.5)(x)

		x = Dense(1)(x)
		x = Activation('relu', name="arousal_output")(x)

		return x

	@staticmethod
	def build(width, height, depth, finalAct="linear"):
	#receive the inputs from the other script (combined-MULT.py)

		#finalAct="relu"
		inputShape = (height, width, depth)
		chanDim = -1

		inputs = Input(shape=inputShape)

		#build both the scene and object categorisers
		sceneBranch = FullNet3.build_scene_branch(inputs, finalAct=finalAct, chanDim=chanDim)
		objectBranch = FullNet3.build_object_branch(inputs, finalAct=finalAct, chanDim=chanDim)
		faceBranch = FullNet3.build_face_branch(inputs, chanDim=chanDim)

		#combined the out puts of each branch of the network

		combinedInput = concatenate([sceneBranch.output, objectBranch.output, faceBranch.output])

		valenceBranch = FullNet3.build_valence_branch(combinedInput, finalAct=finalAct)
		arousalBranch = FullNet3.build_arousal_branch(combinedInput, finalAct=finalAct)


		model = Model(inputs=inputs,
					  outputs=[valenceBranch, arousalBranch],
					  name="fullnet")

		return model


