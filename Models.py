# different models to play around with

from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K

K.set_image_data_format('channels_last')


def SuperMIBI_1(input_image):
    x_input = Input(input_image)

    # first layer
    x = Conv2D(filters=128, kernel_size=(14, 14), strides=(1, 1), padding='same', name='conv1',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x_input)
    x = BatchNormalization(axis=3, name='bn_1')(x)
    x = Activation('relu', name='relu_1')(x)

    # second layer
    #x = Conv2D(filters=64, kernel_size=(20, 20), strides=(1, 1), padding='same', name='conv2',
    #           kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    #x = BatchNormalization(axis=3, name='bn_2')(x)
    #x = Activation('relu', name='relu_2')(x)

    # third layer: 1x1 convolution

    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', name='conv3',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn_3')(x)
    x = Activation('relu', name='relu_3')(x)

    # optional large layer
    #x = Conv2D(filters=5, kernel_size=(20, 20), strides=(1, 1), padding='same', name='conv2',
    #           kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    #x = BatchNormalization(axis=3, name='bn_2')(x)
    #x = Activation('relu', name='relu_2')(x)


    # fourth layer: predict
    x = Conv2D(filters=input_image[2], kernel_size=(5, 5), strides=(1, 1), padding='same', name='conv4',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn_4')(x)
    x = Activation('relu', name='relu_4')(x)

    model = Model(inputs=x_input, outputs=x, name='M1')

    return model


def SuperMIBI_2(input_image):
    x_input = Input(input_image)

    # first layer
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x_input)
    x = BatchNormalization(axis=3, name='bn_1')(x)
    x = Activation('relu', name='relu_1')(x)

    # max pool
    x = MaxPooling2D(pool_size=(2, 2), name='pool_1')(x)

    # second layer
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn_2')(x)
    x = Activation('relu', name='relu_2')(x)

    # max pool
    x = MaxPooling2D(pool_size=(2, 2), name='pool_2')(x)

    # third layer
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn_3')(x)
    x = Activation('relu', name='relu_3')(x)

    # max pool
    #x = MaxPooling2D(pool_size=(2, 2))(x)

    # third layer
    #x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
    #           kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    #x = BatchNormalization(axis=3)(x)
    #x = Activation('relu')(x)
    
    # upsample
    x = UpSampling2D(size=(2, 2), name='up_1')(x)

    # 4th layer
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv4',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn_4')(x)
    x = Activation('relu', name='relu_4')(x)

    # upsample
    x = UpSampling2D(size=(2, 2), name='up_2')(x)

    # 5th layer
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv5',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn_5')(x)
    x = Activation('relu', name='relu_5')(x)

    # upsample
   # x = UpSampling2D(size=(2, 2))(x)

    # 5th layer
    #x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
     #          kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    #x = BatchNormalization(axis=3)(x)
    #x = Activation('relu')(x)

    # 6th layer, 1x1
    x = Conv2D(filters=input_image[2], kernel_size=(1, 1), strides=(1, 1), padding='same', name='conv6',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=3, name='bn_6')(x)
    x = Activation('relu', name='relu_6')(x)

    model = Model(inputs=x_input, outputs=x, name='M2')

    return model
