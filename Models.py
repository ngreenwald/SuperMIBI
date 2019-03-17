# different models to play around with

from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K

K.set_image_data_format('channels_last')


def SuperMIBI_1(input_image):
    x_input = Input(input_image)

    # first layer
    x = Conv2D(filters=128, kernel_size=(9, 9), strides=(1, 1), padding='same', name='conv1',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x_input)
    x = BatchNormalization(axis=3, name='bn_1')(x)
    x = Activation('relu', name='relu_1')(x)

    # second layer
    x = Conv2D(filters=64, kernel_size=(20, 20), strides=(1, 1), padding='same', name='conv2',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn_2')(x)
    x = Activation('relu', name='relu_2')(x)

    # third layer: 1x1 convolution

    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', name='conv3',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn_3')(x)
    x = Activation('relu', name='relu_3')(x)

    # fourth layer: predict

    x = Conv2D(filters=input_image[2], kernel_size=(5, 5), strides=(1, 1), padding='same', name='conv4',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn_4')(x)
    x = Activation('relu', name='relu_4')(x)

    model = Model(inputs=x_input, outputs=x, name='M1')

    return model
