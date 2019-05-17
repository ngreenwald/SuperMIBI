# Gaussian smoothing

from csbdeep.models import CARE
from csbdeep.utils import plot_some, to_color
import h5py
import numpy as np
import re
import matplotlib.pyplot as plt
import sys
import scipy.ndimage as ndimage

input_channels = sys.argv[1:len(sys.argv)]
sigmas = [1,3,5]

# load the data
base_dir = '/home/ubuntu/SuperMIBI/data/'
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

# Specify channels
print('using previously saved channel names for subsetting')
chans = np.load(base_dir + 'chan_names.npy')

keepers = input_channels
keep_idx = np.isin(chans, keepers)
if np.sum(keep_idx) == 0:
    raise ValueError("Did not supply valid channel name")

print('analyzing the following channels: {}'.format(chans[keep_idx]))

x_train, x_test = x_train[:, :, :, keep_idx], x_test[:, :, :, keep_idx]
y_train, y_test = y_train[:, :, :, keep_idx], y_test[:, :, :, keep_idx]


# Determine best sigma using training set
s_mae = []
s_mse = []
for s in sigmas:
    all_mae = []
    all_mse = []
    for i in range(len(x_train)):
        img = x_train[i]
        gaus = ndimage.gaussian_filter(img, sigma=s)
        mae = np.abs(gaus - y_train[i]).mean(axis=None)
        mse = np.square(gaus - y_train[i]).mean(axis=None)
        all_mae.append(mae)
        all_mse.append(mse)
    s_mae.append(np.mean(all_mae))
    s_mse.append(np.mean(all_mse))

ind_mae = s_mae.index(min(s_mae))
ind_mse = s_mse.index(min(s_mse))

sig_mae = sigmas[ind_mae]
sig_mse = sigmas[ind_mse]
print('train_mae best sigma: ' + str(sig_mae))
print('train_mse best sigma: ' + str(sig_mse))

print('train_mae: ' + str(s_mae[ind_mse]))
print('train_mse: ' + str(s_mse[ind_mse]))


# Determine validation error
val_mse = []
val_mae = []
for i in range(len(x_test)):
    img = x_test[i]
    gaus = ndimage.gaussian_filter(img, sigma=sig_mse)
    mae = np.abs(gaus - y_test[i]).mean(axis=None)
    mse = np.square(gaus - y_test[i]).mean(axis=None)
    val_mae.append(mae)
    val_mse.append(mse)
print('val_mae: ' + str(np.mean(val_mae)))
print('val_mse: ' + str(np.mean(val_mse)))


