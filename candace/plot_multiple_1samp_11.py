# Script specifically for 11 channel model because channels not in model name
# epoch1000_11_markers
# Plot combined and individual channels, don't plot scale
# Plot 1 sample per pdf

from csbdeep.models import CARE
from csbdeep.utils import plot_some, to_color
import h5py
import numpy as np
import re
import matplotlib.pyplot as plt
import sys

n = 'epoch1000_11_markers'
models_basedir = '/home/ubuntu/candace/models/'
nsamp = 5

base_dir = '/home/ubuntu/SuperMIBI/data/'
print('loading previously saved data')
# read in data to avoid reprocessing on subsequent analysis
x_data_reload_test = h5py.File(base_dir + 'x_cropped_test.h5', 'r')
y_data_reload_test = h5py.File(base_dir + 'y_cropped_test.h5', 'r')
x_test = x_data_reload_test['x_test'][:]
y_test = y_data_reload_test['y_test'][:]

print('using previously saved channel names for subsetting')
chans = np.load(base_dir + 'chan_names.npy')
chans_lower = np.array([x.lower() for x in chans])

# For 11 channel model, channels are not in model name, so provide them here
keepers = ['CD45.tif', 'CD20.tif', 'CD3.tif', 'CD4.tif', 'CD8.tif', 'PD-1.tif', 'CD14.tif', 'HLA-Class-1.tif', 'NaKATPase.tif', 'Vimentin.tif', 'dsDNA.tif']
keep_idx = np.isin(chans, keepers)

print('analyzing the following channels: {}'.format(chans[keep_idx]))
x_test_i = x_test[:, :, :, keep_idx]
y_test_i = y_test[:, :, :, keep_idx]

nch = len(chans[keep_idx])

# Fit the model
print('analyzing the model: '+n)
model = CARE(config=None, name=n, basedir=models_basedir)

h = 3 #1 row for input, 1 row for ground truth, 1 row for output
w = nch+1 #1 column for combined, 1 column for each marker

for s in range(nsamp):

    _P = model.keras_model.predict(x_test_i[np.newaxis, s, :, :, :])
    _P_mean  = _P[...,:(_P.shape[-1]//2)]

    fig = plt.figure(figsize=(80,30))
    plt.suptitle(n + ", Sample " + str(s), fontsize=30)
    plt.gcf().text(0.08, 0.76, "Input", fontsize=24)
    plt.gcf().text(0.08, 0.5, "Ground truth", fontsize=24)
    plt.gcf().text(0.08, 0.22, "Output", fontsize=24)

    # Plot combined input
    plt.subplot(h, w, 1)
    plt.title('Combined', fontsize=24)
    img = x_test_i[np.newaxis, s, :, :, :]
    img = np.squeeze(np.stack(map(to_color,img)))
    plt.imshow(img)
    plt.axis("off")

    # Plot combined ground truth
    plt.subplot(h, w, w+1)
    img = y_test_i[np.newaxis, s, :, :, :]
    img = np.squeeze(np.stack(map(to_color,img)))
    plt.imshow(img)
    plt.axis("off")

    # Plot combined output
    plt.subplot(h, w, 2*w+1)
    img = _P_mean
    img = np.squeeze(np.stack(map(to_color,img)))
    plt.imshow(img)
    plt.axis("off")

    for k in range(len(chans[keep_idx])):
        c = chans[keep_idx][k]
        
        # Plot single channel input
        plt.subplot(h, w, k+2)
        plt.title(c, fontsize=24)
        img =  x_test_i[np.newaxis, s, :, :, k]
        img = np.squeeze(img)
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        
        # Plot single channel ground truth
        plt.subplot(h, w, w+k+2)
        img = y_test_i[np.newaxis, s, :, :, k]
        img = np.squeeze(img)
        plt.imshow(img, cmap='gray')
        plt.axis("off")

        # Plot single channel output
        plt.subplot(h, w, 2*w+k+2)
        img = _P_mean[:, :, :, k]
        img = np.squeeze(img)
        plt.imshow(img, cmap='gray')
        plt.axis("off")

    fig.savefig('/home/ubuntu/candace/plots/individual_samps_multiple/' + n + '_' + str(s) + '.pdf')

