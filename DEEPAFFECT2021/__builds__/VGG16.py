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
from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.layers import concatenate

class vgg16:
    @staticmethod
    def build (width, height, depth, finalAct="linear"):

        inputShape = (height, width, depth)
        # chanDim = -1
        inputs = Input(shape=inputShape)

        vgg16 = VGG19(weights='imagenet', include_top=True, input_shape=(224, 224, 3))


        x = Dense(1, activation='relu', name='predictions')(vgg16.layers[-2].output)


        # x = Flatten()(x)
        # x = Dense(4096, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(4096, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(1, activation='linear')(x)

        model = Model(vgg16.input, x)

        return model

    @staticmethod
    def indiv(width, height, depth, finalAct):
        inputShape = (height, width, depth)
        # chanDim = -1
        inputs = Input(shape=inputShape)

        base_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

        # for layer in base_model.layers:
        #     layer.trainable = False

        x = base_model(inputs)

        x = Dense(3000, activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(1000, activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(500, activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(1)(x)
        x = Activation('relu', name="arousal_output")(x)

        model = Model(inputs, x)

        return model

    def duo(width, height, depth, finalAct="linear"):

        inputShape = (height, width, depth)
        # chanDim = -1
        inputs = Input(shape=inputShape)

        vgg16 = VGG19(weights='imagenet', include_top=True, input_shape=(224, 224, 3))


        x = Dense(1, activation='relu', name="arousal_output")(vgg16.layers[-2].output)
        y = Dense(1, activation='relu', name="valence_output")(vgg16.layers[-2].output)

        model = Model(inputs=vgg16.input,
                      outputs=[x, y],
                      name="vgg")

        return model







        # if we are using "channels first", update the input shape
        # and channels dimension
        #if K.image_data_format() == "channels_first":
            #inputShape = (depth, height, width)
            #chanDim = 1

        # conv_base = VGG16(weights = 'imagenet',
        #                   include_top=False,
        #                   input_shape=inputShape)
        #
        # model.add(conv_base)
        # model.add(layers.Flatten())
        # model.add(layers.Dense(3000))
        # model.add(Activation("relu"))
        # model.add(Dropout(0.5))
        #
        # model.add(layers.Dense(1000))
        # model.add(Activation("relu"))
        # model.add(Dropout(0.5))
        # model.add(layers.Dense(1000))
        # model.add(Activation("relu"))
        # model.add(Dropout(0.5))
        #
        # model.add(Dense(1))
        # model.add(Activation("linear"))
        #
        # return model

