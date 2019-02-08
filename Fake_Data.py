# Generate a couple different fake datasets
import numpy as np
import scipy
import skimage.draw
import matplotlib.pyplot as plt


# first create a number of rectangles and ellipses
# each image will be in a 64x64 box
# The min radius will be 5, and the maximum radius will be 15. Hence the center of the elipse must be 15<coord<49


def squares_and_ellipses():
    example_num = 300
    data = np.zeros((example_num * 2, 64, 64, 1))
    labels = np.repeat([0, 1], example_num)
    np.random.seed(1)

    row_coord = np.random.sample(example_num) * 32 + 16
    col_coord = np.random.sample(example_num) * 32 + 16
    row_rad = np.random.sample(example_num) * 10 + 5
    col_rad = np.random.sample(example_num) * 10 + 5
    rotation_el = np.random.sample(example_num) * 360

    for i in range(example_num):
        row_coords, col_coords = skimage.draw.ellipse(int(row_coord[i]), int(col_coord[i]), int(row_rad[i]),
                                                      int(col_rad[i]), rotation=np.deg2rad(rotation_el[i]))
        data[i, row_coords, col_coords, :] = 1


    # Each rectangle will have lengths between 5 and 15. Its upper left hand corner will be between 0 and 48,48

    corner_row = np.random.sample(example_num) * 48
    corner_col = np.random.sample(example_num) * 48
    row_len = np.random.sample(example_num) * 10 + 5
    col_len = np.random.sample(example_num) * 10 + 5

    for i in range(example_num):
        row_coords, col_coords = skimage.draw.rectangle((int(corner_row[i]), int(corner_col[i])),
                                                        extent=(int(row_len[i]), int(col_len[i])))
        data[i + example_num, row_coords, col_coords, :] = 1

    # create a train/test split
    train_idx = np.random.binomial(1, 0.9, example_num * 2) == 1
    return data[train_idx, :, :, :], labels[train_idx], data[~train_idx, :, :, :], labels[~train_idx]

