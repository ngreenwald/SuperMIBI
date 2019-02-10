# Generate a couple different fake datasets
import numpy as np
import skimage.draw


# first create a number of rectangles and ellipses
# each image will be in a 64x64 box
# The min radius will be 5, and the maximum radius will be 15. Hence the center of the elipse must be 15<coord<49

# TODO change to random.randint for clarity

def squares_and_ellipses(example_num=300):
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


def low_and_hi_res(example_num, base_value, dropout_val):
    """ Generates fake data that aproximates downsampling in time for a set of images
        Draws filled circles and rectangles of various shapes. Each shape has values ranging from 0 to base_value,
        whereas the downsampled version will have these values reduced by a random draw between 0 and dropout value
        independently for each pixel. Currently does not have a noise aspect"""
    np.random.seed(2)
    # generate random squares and ellipses from previous function
    train_temp, _, test_temp, _ = squares_and_ellipses(example_num)
    data = np.append(train_temp, test_temp, axis=0)

    # Generate a mask that will scale up the values in the true data stochastically
    scale_mask = np.random.sample(np.size(data)) * base_value
    scale_mask = np.reshape(scale_mask, (np.shape(data)))
    scale_mask = np.floor(scale_mask)

    data = data * scale_mask

    dropout_mask = np.random.sample(np.size(data)) * dropout_val
    dropout_mask = np.reshape(dropout_mask, data.shape)
    dropout_mask = np.floor(dropout_mask)

    down_sampled_data = np.copy(data)
    dowm_sampled_data = down_sampled_data - dropout_mask
    dowm_sampled_data = np.floor(dowm_sampled_data)
    negative_mask = dowm_sampled_data < 0
    dowm_sampled_data[negative_mask] = 0

    train_idx = train_temp.shape[0]
    return (dowm_sampled_data[:train_idx, :, :, :], data[:train_idx, :, :, :], dowm_sampled_data[train_idx:, :, :, :],
            data[train_idx:, :, :, :])


