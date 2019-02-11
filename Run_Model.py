import Data_Import
import Models
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# load the data
x_dirs = ['path1', 'path2', 'path3']
y_dirs = ['path1', 'path2', 'path3']
channels = ['dsDNA.tif', 'LaminAC.tif']
x_data, y_data = Data_Import.load_dataset(x_dirs, y_dirs, channels)

# crop data
x_data_cropped = Data_Import.crop_dataset(dataset=x_data, crop_size=128, stride_fraction=0.333)
y_data_cropped = Data_Import.crop_dataset(dataset=y_data, crop_size=128, stride_fraction=0.333)


# create train and test split
np.random.seed(seed=1)
choo_choo_idx = np.random.binomial(n=1, p=0.01, size=x_data_cropped.shape[0])
x_train, x_test = x_data_cropped[choo_choo_idx, :, :, :], x_data_cropped[~choo_choo_idx, :, :, :]
y_train, y_test = y_data_cropped[choo_choo_idx, :, :, :], y_data_cropped[~choo_choo_idx, :, :, :]


# create model
model_1 = Models.SuperMIBI_1((128, 128, 30))
model_1.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# initializer data augmentation iterator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format='channels_last',
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)


# Fit the model on the batches generated by datagen.flow().
model_1.history = model_1.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=10)


# evaluate model
preds = model_1.evaluate(x=x_test, y=y_test)
print()
print("Training accuracy: {}".format(np.round(model_1.history.history['acc'], 2)))
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))