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


def Face (inputs):

    Face = VGG19(include_top=True, weights='imagenet', input_shape=(224, 224, 3))

    # for layer in Face.layers: #[:17]
    #     layer.trainable = False

    # x = Face(inputs)
    #
    # x = Flatten()(x)

    x = Dense(2048, trainable=True)(Face.layers[-4].output)
    x = Activation("relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    x = Dense(1024, trainable=True)(x)
    x = Activation("relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    x = Dense(512, trainable=True)(x)
    x = Activation("relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    # softmax classifier
    x = Dense(5, trainable=True)(x)
    x = Activation("softmax")(x)

    model = Model(Face.input, x, name='Face-net')

    #model.load_weights("Full5BestWEIGHTS-2.h5")  # CHANGE NODES
    # model.load_weights("VGG19_3_EarlyStop.h5")
    ### aa9EmotionBest.h5
    # tt_EarlyStop.h5

    return model

