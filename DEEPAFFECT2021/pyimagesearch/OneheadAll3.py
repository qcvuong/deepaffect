# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
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
from keras.applications.inception_v3 import InceptionV3
from keras.layers import concatenate
from keras import layers

class ClassifyNet2:
	@staticmethod
	def build_face_branch(inputs, chanDim=-1):

		Face = VGG19(include_top=False, weights=None, input_shape=(224, 224, 3))

		# for layer in Face.layers:
		# 	layer.trainable = False


		x = Face (inputs)

		x = Flatten()(x)

		x = Dense(2048, trainable=True)(x)
		x = Activation("relu")(x)
		#x = BatchNormalization()(x)
		#x = Dropout(0.5)(x)

		x = Dense(1024, trainable=True)(x)
		x = Activation("relu")(x)
		#x = BatchNormalization()(x)
		#x = Dropout(0.5)(x)

		x = Dense(512, trainable=True)(x)
		x = Activation("relu")(x)
		#x = BatchNormalization()(x)
		#x = Dropout(0.5)(x)

		# softmax classifier
		x = Dense(9, trainable=True)(x) ##!!
		x = Activation("softmax")(x)

		model = Model(inputs, x)

		model.load_weights("aa9EmotionBest.h5") ##NUEONE

		return model

	@staticmethod
	def build_scene_branch(inputs, finalAct="linear", chanDim = -1):
		# initialize the scene branch and provide input shape, assuming
		# TensorFlow/channels-last ordering

		VGG_plc = VGG16_Places365(weights='places', include_top=True, input_shape=(224, 224, 3))
		# define a branch for the scene clasification - check these are doing what they say they are doing

		# for layer in VGG_plc.layers:
		# 	layer.trainable = False

		x = VGG_plc (inputs)

		model = Model(inputs, x)

		return model

	@staticmethod
	def build_object_branch (inputs, finalAct="linear", chanDim = -1):

		base_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

		# for layer in base_model.layers:
		# 	layer.trainable = False

		x = base_model (inputs)

		model = Model(inputs, x)

		return model


	@staticmethod
	def build(width, height, depth, finalAct="linear"):
	#receive the inputs from the other script (combined-MULT.py)

		#finalAct="relu"
		inputShape = (height, width, depth)
		chanDim = -1

		inputs = Input(shape=inputShape)

		#build both the scene and object categorisers
		sceneBranch = ClassifyNet2.build_scene_branch(inputs, finalAct=finalAct, chanDim=chanDim)
		objectBranch = ClassifyNet2.build_object_branch(inputs, finalAct=finalAct, chanDim=chanDim)
		faceBranch = ClassifyNet2.build_face_branch(inputs, chanDim=chanDim)
		#combined the out puts of each branch of the network

		combinedInput = concatenate([sceneBranch.output, objectBranch.output, faceBranch.output])

		# output = objectBranch.output

		#use the combined output as the input for a fully connected head
		#leading to a single contineous value as the output


		# x = Dense(1000, activation='relu')(output)
		# #x = Dropout(0.5)(x)

		x = Dense(3000, activation='relu')(combinedInput)
		x = Dropout(0.5)(x)

		x = Dense(1000, activation='relu')(x)
		x = Dropout(0.5)(x)

		x = Dense(500, activation='relu')(x)
		x = Dropout(0.5)(x)

		x = Dense(1, activation='relu')(x)

		model = Model(inputs=inputs,
					  outputs=x,
					  name="classifynet")

		return model


