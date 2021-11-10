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
from keras.layers.core import Lambda



def FaceNet(inputs):
    x = VGG19(include_top=False, weights=None, input_shape=(224, 224, 3))

    x = Flatten()(x)

    x = Dense(2048)(x)
    x = Activation("relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    x = Dense(1024)(x)
    x = Activation("relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    x = Dense(512)(x)
    x = Activation("relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)

    # softmax classifier
    x = Dense(9)(x)
    x = Activation("softmax")(x)

    model = Model(inputs, x)

    # model.load_weights("VGG19_3_EarlyStop.h5")
    # model.layers.trainable = False

    return model



class xxx:
    @staticmethod
    def xx(width, height, depth, classes, finalAct):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
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

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        model.load_weights("best_face_model_weights.h5")

        model.add(Dense(1))
        model.add(Activation("linear"))

        # return the constructed network architecture
        return model

