# example to see how tf.keras works
import numpy as np
import tensorflow as tf
import tensorflow.keras


from tensorflow.keras import layers
from tensorflow.keras.layers import Input, ZeroPadding2D
from tensorflow.keras.models import Model


# generate data



def simple_model(input_image):
    x_input = Input(input_image)
    x = ZeroPadding2D((3,3))(x_input)

    return x
    # model = Model(x = x_input, y = x, name = 'first_model')
    # return model


new = simple_model((5,5,2))



