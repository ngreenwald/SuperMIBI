# example to see how tf.keras works
import numpy as np
from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D, Activation, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import glorot_uniform
K.set_image_data_format('channels_last')



import Fake_Data
import importlib


def simple_model(input_image):
    X_input = Input(input_image)
    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(filters=10, kernel_size=(5, 5), strides=(1, 1), padding='valid', name='conv_1',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(X)
    X = BatchNormalization(axis=3, name='bn_1')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D(pool_size=(2, 2), name='pool_1')(X)

    X = Flatten()(X)
    X = Dense(units=1, activation='sigmoid', name='fc', kernel_regularizer=regularizers.l2(0.01))(X)

    model = Model(inputs=X_input, outputs=X, name='first_model')
    return model


new_model = simple_model((64, 64, 1))


importlib.reload(Fake_Data)
X_train, Y_train, X_test, Y_test = Fake_Data.squares_and_ellipses()

new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = new_model.fit(x=X_train, y=Y_train, epochs=10, batch_size=20)

preds = new_model.evaluate(x=X_test, y=Y_test)
print()
print("Training accuracy: {}".format(np.round(history.history['acc'], 2)))
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))


