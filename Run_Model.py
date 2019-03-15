import Data_Import
import Models
import numpy as np
import os
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys


# check for command line arguments for model name

if len(sys.argv) < 4:
    raise ValueError("did not include enough  command line arguments. Syntax: python Run_Model.py model_name num_epochs channel_name1.tif...")
else:
    model_name = sys.argv[1]
    num_epochs = int(sys.argv[2])
    input_channels = sys.argv[3:len(sys.argv)]

print("using {} as model name".format(model_name))
print("training for {} epochs".format(num_epochs))
print("using {} as channels".format(input_channels))
# load the data

#base_dir = '/Users/noahgreenwald/Documents/MIBI_Data/CS230/test_run/'
base_dir = '/home/ubuntu/SuperMIBI/data/test_run/'

if os.path.isfile(base_dir + 'x_cropped_train.h5'):
    print('loading previously saved data')
    # read in data to avoid reprocessing on subsequent analysis
    x_data_reload_train = h5py.File(base_dir + 'x_cropped_train.h5', 'r')
    x_data_reload_test = h5py.File(base_dir + 'x_cropped_test.h5', 'r')
    y_data_reload_train = h5py.File(base_dir + 'y_cropped_train.h5', 'r')
    y_data_reload_test = h5py.File(base_dir + 'y_cropped_test.h5', 'r')
    x_train = x_data_reload_train['x_train'][:]
    x_test = x_data_reload_test['x_test'][:]
    y_train = y_data_reload_train['y_train'][:]
    y_test = y_data_reload_test['y_test'][:]

else:
    print('no previous data data detected. Loading from directory')
    x_dirs = ['Point7/', 'Point9/', 'Point11/']
    y_dirs = ['Point8/', 'Point10/', 'Point12/']
    dir_suffix = 'TIFs/'

    x_dirs = [base_dir + x + dir_suffix for x in x_dirs]
    y_dirs = [base_dir + y + dir_suffix for y in y_dirs]
    channels = os.listdir(x_dirs[0])
    bad_bois = ['Background.tif', 'Fe.tif', 'Ta.tif', 'Au.tif', 'Ca.tif', 'totalIon.tif', 'Na.tif', 'C.tif', 'Si.tif']
    channels = [x for x in channels if x not in bad_bois]

    # get dat data
    x_data, y_data = Data_Import.load_dataset(x_dirs, y_dirs, channels)

    # normalize data on a per channel basis
    x_data= Data_Import.data_norm(x_data)
    y_data= Data_Import.data_norm(y_data)

    # crop data
    print('cropping the data')
    x_data_cropped = Data_Import.crop_dataset(dataset=x_data, crop_size=128, stride_fraction=0.333)
    y_data_cropped = Data_Import.crop_dataset(dataset=y_data, crop_size=128, stride_fraction=0.333)

    # save data for quicker loading next time
    
    # create train and test split
    print('Generating train/test split')
    np.random.seed(seed=1)
    choo_choo_idx = np.random.binomial(n=1, p=0.95, size=x_data_cropped.shape[0])
    # choo_choo_idx = random.sample(range(x_data_cropped.shape[0]), np.floor(x_data_cropped.shape[0] * 0.95))
    
    idx_num = np.arange(x_data_cropped.shape[0])
    np.random.shuffle(idx_num)

    # change to boolean so that indexing works
    choo_choo_idx = choo_choo_idx == 1
    train_idx = idx_num[choo_choo_idx]
    test_idx =idx_num[~choo_choo_idx]
    x_train, x_test = x_data_cropped[train_idx, :, :, :], x_data_cropped[test_idx, :, :, :]
    y_train, y_test = y_data_cropped[train_idx, :, :, :], y_data_cropped[test_idx, :, :, :]
    print('x_train shape is {}. y_train_shape is {}.'.format(x_train.shape, y_train.shape)) 

    print('saving data for faster loading next time')
    h5_x_cropped_train = h5py.File(base_dir + 'x_cropped_train.h5', 'w')
    h5_x_cropped_test = h5py.File(base_dir + 'x_cropped_test.h5', 'w')

    h5_y_cropped_train = h5py.File(base_dir + 'y_cropped_train.h5', 'w')
    h5_y_cropped_test = h5py.File(base_dir + 'y_cropped_test.h5', 'w')
    
    h5_x_cropped_train.create_dataset('x_train', data=x_train)
    h5_x_cropped_test.create_dataset('x_test', data=x_test)
    h5_y_cropped_train.create_dataset('y_train', data=y_train)
    h5_y_cropped_test.create_dataset('y_test', data=y_test)

    h5_x_cropped_train.close()
    h5_x_cropped_test.close()
    h5_y_cropped_train.close()
    h5_y_cropped_test.close()

    chans = np.array(channels)
    np.save(base_dir + 'chan_names', chans)



# include only a subset of channels
if os.path.isfile(base_dir + 'chan_names.npy'):
    print('using previously saved channel names for subsetting')
    chans = np.load(base_dir + 'chan_names.npy')
else:
    print('using currently generated channel names')

#keepers = ['H3K9Ac.tif']
keepers = input_channels
keep_idx = np.isin(chans, keepers)
print('analyzing the following channels: {}'.format(chans[keep_idx]))

x_train, x_test = x_train[:, :, :, keep_idx], x_test[:, :, :, keep_idx]
y_train, y_test = y_train[:, :, :, keep_idx], y_test[:, :, :, keep_idx]
print('new x_train shape is {}. new y_train_shape is {}.'.format(x_train.shape, y_train.shape)) 


# create model
model_1 = Models.SuperMIBI_1((128, 128, np.sum(keep_idx)))
model_1.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])


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
print('training the model')
model_1.history = model_1.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=num_epochs)

# evaluate model
print('evaluating the model')
preds = model_1.evaluate(x=x_test, y=y_test)
print()
print("Training MSE: {}".format(np.round(model_1.history.history['mean_squared_error'], 2)))
print("Loss = " + str(preds[0]))
print("Test MSE = " + str(preds[1]))

print('saving model')
model_1.save('output/models/' + model_name)
np.save('output/models/' + model_name + '_channel_idx.npy', keep_idx)
