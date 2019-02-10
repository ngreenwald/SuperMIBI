# File for importing and organizing the training data
import numpy as np
import math

import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf

def crop_helper(img, crop_size):
    """"Helper function to take an image, and return crops of size crop_size

        Inputs
        img (np.array): A 3D numpy array of shape(rows, columns, channels)
        crop_size (int): Size of the crop to take from the image. Assumes square crops

        Outputs
        cropped_images (np.array): A 4D numpy array of shape (crops, rows, columns, channels)"""

    if len(img.shape) != 3:
        raise ValueError("Incorrect dimensions of input image. Expecting 3D, got {}".format(img.shape))

    # First determine number of crops across the image. Includes last full crop only
    crop_num_row = math.floor(img.shape[0] / crop_size)
    # print("crop_num_row {}".format(crop_num_row))
    crop_num_col = math.floor(img.shape[1] / crop_size)
    # print("crop_num_col {}".format((crop_num_col)))
    cropped_images = np.zeros((crop_num_row * crop_num_col, crop_size, crop_size, img.shape[2]))

    # iterate through the image row by row, cropping based on identified threshold
    img_idx = 0
    for row in range(crop_num_row):
        for col in range(crop_num_col):
            cropped_images[img_idx, :, :, :] = img[(row * crop_size):((row + 1) * crop_size),
                                                   (col * crop_size):((col + 1) * crop_size), :]
            img_idx += 1

    return cropped_images


def crop_image(img, crop_size, stride_fraction):
    """Function to generate a series of tiled crops across an image. The tiled crops will overlap each other, with the
       overlap between tiles determined by the stride fraction. A stride fraction of 0.333 will move the window over
       1/3 of the crop_size in x and y at each step, resulting in 9 distinct crops of the image.

        Inputs
        img (np.array): A 3D numpy array of shape(rows, columns, channels)
        crop_size (int): size of the crop to take from the image. Assumes square crops
        stride_fraction (float): the relative size of the stride for overlapping crops as a function of
        the crop size.

        Outputs
        cropped_images (np.array): A 4D numpy array of shape(crops, rows, cols, channels)"""

    if len(img.shape) != 3:
        raise ValueError("Incorrect dimensions of input image. Expecting 3D, got {}".format(img.shape))

    if crop_size > img.shape[0]:
        raise ValueError("Invalid crop size: img shape is {} and crop size is {}".format(img.shape, crop_size))

    if stride_fraction > 1:
        raise ValueError("Invalid stride fraction. Must be less than 1, passed a value of {}".format(stride_fraction))

    # Determine how many distinct grids will be generated across the image
    stride_step = math.floor(crop_size * stride_fraction)
    num_strides = math.floor(1 / stride_fraction)

    for row_shift in range(num_strides):
        for col_shift in range(num_strides):

            if row_shift == 0 and col_shift == 0:
                # declare data holder
                cropped_images = crop_helper(img, crop_size)
            else:
                # crop the image by the shift prior to generating grid of crops
                img_shift = img[(row_shift * stride_step):, (col_shift * stride_step):, :]
                # print("shape of the input image is {}".format(img_shift.shape))
                temp_images = crop_helper(img_shift, crop_size)
                cropped_images = np.append(cropped_images, temp_images, axis=0)

    return cropped_images

# for checking that the above are working
xx, yy = skimage.draw.ellipse_perimeter(25, 25, 20, 20)
xx1, yy1 = skimage.draw.ellipse_perimeter(25, 25, 15, 15)
xx2, yy2 = skimage.draw.rectangle((20, 20), extent=(10,10))

x = np.zeros((50, 50, 1))
x[xx, yy, 0] = 1
x[xx1, yy1, 0] = 1
x[xx2, yy2, 0] = 1

x = x[:, 6:, :]
plt.imshow(x[:, :, 0])

y = crop_helper(x, 16)

y = crop_image(x, 16, 0.3333)

plt.imshow(y[0, :, :, 0])
x1 = 10

np.zeros((x1 * x1, x.shape[0]))


def load_tifs(dir, channels, dim=1024):
    """Loads tifs from a directory

    Inputs
    dir (string): name of directory to load from
    channels (list): list of channel names
    dim (int): dimension of the images

    Outputs
    data (np.array): array of shape (rows, cols, channels) containing imaging data for that directory"""

    data = np.zeros((dim, dim, len(channels)))
    for idx, chan in enumerate(channels):
        data[:, :, idx] = skimage.io.imread(os.path.join(dir, chan))

    return data


def load_dataset(x_dirs, y_dirs, channels):
    """Loads imaging data from a list of matched x and y points.

    Inputs
    x_dirs (list): directories containing tifs
    y_dirs (list): directories containing tifs
    channels (list): list of channel names

    Outputs
    x_data (np.array): array of shape (num_x_points, rows, cols, num_channels) corresponding to x data
    y_data (np.array): array of shape (num_y_points, rows_cols, num_channels) corresponding to y data"""

    if len(x_dirs) != len(y_dirs):
        raise ValueError("x and y directories do not contain the same number of points: "
                         "x_dirs {}, y_dirs {}".format(len(x_dirs), len(y_dirs)))

    # load x data
    for i in range(len(x_dirs)):

        if i == 0:
            # initialize array
            x_temp = load_tifs(x_dirs[i], channels)
            x_data = np.zeros((len(x_dirs), x_temp.shape[0], x_temp.shape[1], x_temp.shape[2]))
            x_data[i, :, :, :] = x_temp

        else:
            x_temp = load_tifs(x_dirs[i], channels)
            x_data[i, :, :, :] = x_temp

    # load y data
    for j in range(len(y_dirs)):

        if j == 0:
            # initialize array
            y_temp = load_tifs(y_dirs[j], channels)
            y_data = np.zeros((len(y_dirs), y_temp.shape[0], y_temp.shape[1], y_temp.shape[2]))
            y_data[j, :, :, :] = y_temp

        else:
            y_temp = load_tifs(y_dirs[i], channels)
            y_data[j, :, :, :] = y_temp

    return x_data, y_data


def crop_dataset(dataset, crop_size, stride_fraction):
    """Function to take a dataset and return series of tiled crops.

    Inputs
    dataset (np.array): a 4D array of shape(points, rows, cols, channels) that contains the imaging data
    crop_size (int): size of crops to take from each image
    stride fraction (float): fraction of crop_size to move over tiling window each time to generate semi-overlapping
    crops

    Outputs
    cropped_dataset: a 4D array of shape (crops, rows, cols, channels) that contains the cropped data"""

    if len(dataset.shape) != 4:
        raise ValueError("Invalid dimensions for dataset. Expecting 4, found {}".format(dataset.shape))

    for i in range(dataset.shape[0]):
        if i == 0:
            # initialize the array
            temp_dataset = crop_image(dataset[i, :, :, :], crop_size, stride_fraction)
            num_crops = temp_dataset.shape[0]
            cropped_dataset = np.zeros((dataset.shape[0] * num_crops, temp_dataset.shape[1],
                                        temp_dataset.shape[2], temp_dataset.shape[3]))

            # index into array and store values
            cropped_dataset[(num_crops * i):(num_crops * (i + 1)), :, :, :] = temp_dataset
        else:
            temp_dataset = crop_image(dataset[i, :, :, :], crop_size, stride_fraction)
            cropped_dataset[(num_crops * i):(num_crops * (i + 1)), :, :, :] = temp_dataset

    return cropped_dataset


dirs = ['/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation/Contours/First_Run/Point18/TIFsNoNoise/',
        '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation/Contours/First_Run/Point12/TIFsNoNoise/']

channels = ['dsDNA.tif', 'Na.tif', 'H3K9Ac.tif']


x, y = load_dataset(dirs, dirs, channels)
x_cropped = crop_dataset(x, 128, 0.5)


x = skimage.io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation/Contours/First_Run/Point1/TIFsNoNoise/dsDNA.tif')

temp = np.zeros((2, 50, 50, 1))
temp[0, :, :, :] = x
temp[1, :, :, :] = x * 2


y1 = crop_dataset(temp, 16, 1)
y = crop_image(temp[0, :, :, :], 16, 1)