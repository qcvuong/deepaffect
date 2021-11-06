# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras import layers
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
from keras.layers import concatenate

class inception:
    @staticmethod
    def build (width, height, depth, finalAct="linear"):

        model = Sequential()
        inputShape = (height, width, depth)
        #chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        #if K.image_data_format() == "channels_first":
            #inputShape = (depth, height, width)
            #chanDim = 1

        conv_base = InceptionV3(weights = 'imagenet',
                          include_top=False,
                          input_shape=inputShape)

        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(3000))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(layers.Dense(1000))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(layers.Dense(1000))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(1))
        model.add(Activation("linear"))

        return model

