# Plot combined and individual channels, don't plot scale
# Plot 1 sample per pdf
# For models with multiple inputs but 1 output
# python plot_multiple_1samp_1output.py model_name output_channel input_channel1 input_channel2 ... 

from csbdeep.models import CARE
from csbdeep.utils import plot_some, to_color
import h5py
import numpy as np
import re
import matplotlib.pyplot as plt
import sys

n = sys.argv[1]
output_channels = sys.argv[2]
input_channels = sys.argv[3:len(sys.argv)]

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

keepers_in = input_channels
keep_idx_in = np.isin(chans, keepers_in)
if np.sum(keep_idx_in) == 0:
    raise ValueError("Did not supply valid channel name")

keepers_out = output_channels
keep_idx_out = np.isin(chans, keepers_out)
if np.sum(keep_idx_out) == 0:
    raise ValueError("Did not supply valid channel name")

print('analyzing the following in channels: {}'.format(chans[keep_idx_in]))
print('analyzing the following out channels: {}'.format(chans[keep_idx_out]))

x_test_i = x_test[:, :, :, keep_idx_in]
y_test_i = y_test[:, :, :, keep_idx_out]

nch = len(chans[keep_idx_in])

# Fit the model
print('analyzing the model: '+n)
model = CARE(config=None, name=n, basedir=models_basedir)

h = 3 #1 row for input, 1 row for ground truth, 1 row for output
w = nch #1 column for each input marker

for s in range(nsamp):

    _P = model.keras_model.predict(x_test_i[np.newaxis, s, :, :, :])
    _P_mean  = _P[...,:(_P.shape[-1]//2)]

    fig = plt.figure(figsize=(30,30))
    plt.suptitle(n + ", Sample " + str(s), fontsize=30)
    plt.gcf().text(0.02, 0.76, "Input", fontsize=24)
    plt.gcf().text(0.02, 0.5, "Ground truth", fontsize=24)
    plt.gcf().text(0.02, 0.22, "Output", fontsize=24)

    # Plot the one channel ground truth and output
    plt.subplot(h, w, w+1)
    img = y_test_i[np.newaxis, s, :, :, :]
    img = np.squeeze(img)
    plt.imshow(img, cmap='gray')
    plt.axis("off")

    plt.subplot(h, w, 2*w+1)
    img = _P_mean
    img = np.squeeze(img)
    plt.imshow(img, cmap='gray')
    plt.axis("off")
 
    for k in range(len(chans[keep_idx_in])):
        c = chans[keep_idx_in][k]
        if c==output_channels:
            plt.subplot(h, w, 1)
            plt.title(c, fontsize=24)
            img =  x_test_i[np.newaxis, s, :, :, k]
            img = np.squeeze(img)
            plt.imshow(img, cmap='gray')
            plt.axis("off")
        else: 
           # Plot single channel input
            plt.subplot(h, w, k+2)
            plt.title(c, fontsize=24)
            img =  x_test_i[np.newaxis, s, :, :, k]
            img = np.squeeze(img)
            plt.imshow(img, cmap='gray')
            plt.axis("off")
     
    fig.savefig('/home/ubuntu/candace/plots/individual_samps_multiple/' + n + '_' + str(s) + '.pdf')

