# Plot results from different channels
# Plot both channels and individual channels, don't plot scale
# Plot mulitple samples in 1 pdf

from csbdeep.models import CARE
from csbdeep.utils import plot_some, to_color
import h5py
import numpy as np
import re
import matplotlib.pyplot as plt
import sys

n = sys.argv[1]
models_basedir = '/home/ubuntu/candace/models/'

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

nsamp = 5 

# Assumes same channels for all the models
ch = [re.search('_(.+)_',n).group(1), re.search('_.+_(.+)$',n).group(1)]
keepers = [s + '.tif' for s in ch]
keepers_lower = [x.lower() for x in keepers]
keep_idx = np.isin(chans_lower, keepers_lower)
print('analyzing the following channels: {}'.format(chans[keep_idx]))
x_test_i = x_test[:, :, :, keep_idx]
y_test_i = y_test[:, :, :, keep_idx]

# Fit the model
print('analyzing the model: '+n)
model = CARE(config=None, name=n, basedir=models_basedir)

fig = plt.figure(figsize=(30,30))
plt.gcf().text(0.45, 0.92, n, fontsize=30)
plt.gcf().text(0.05, 0.835, "Input", fontsize=16)
plt.gcf().text(0.05, 0.75, "Ground truth", fontsize=16)
plt.gcf().text(0.05, 0.665, "Output", fontsize=16)
plt.gcf().text(0.05, 0.58, chans[keep_idx][0]+ " input", fontsize=16)
plt.gcf().text(0.05, 0.495, chans[keep_idx][0]+ " ground truth", fontsize=16)
plt.gcf().text(0.05, 0.4, chans[keep_idx][0]+ " output", fontsize=16)
plt.gcf().text(0.05, 0.32, chans[keep_idx][1]+ " input", fontsize=16)
plt.gcf().text(0.05, 0.23, chans[keep_idx][1]+ " ground truth", fontsize=16)
plt.gcf().text(0.05, 0.14, chans[keep_idx][1]+ " output", fontsize=16)


h = 3*(len(keepers)+1) #for combined and individual channels, 1 row for input, 1 row for ground truth, 1 row for output
w = nsamp

for s in range(nsamp):

    _P = model.keras_model.predict(x_test_i[np.newaxis, s, :, :, :])
    _P_mean  = _P[...,:(_P.shape[-1]//2)]

    # Plot combined input
    plt.subplot(h, w, s+1)
    plt.title("Sample " + str(s+1), fontsize=24)
    img = x_test_i[np.newaxis, s, :, :, :]
    img = np.squeeze(np.stack(map(to_color,img)))
    plt.imshow(img)
    plt.axis("off")

    # Plot combined ground truth
    plt.subplot(h, w, w+s+1)
    img = y_test_i[np.newaxis, s, :, :, :]
    img = np.squeeze(np.stack(map(to_color,img)))
    plt.imshow(img)
    plt.axis("off")

    # Plot combined output
    plt.subplot(h, w, 2*w+s+1)
    img = _P_mean
    img = np.squeeze(np.stack(map(to_color,img)))
    plt.imshow(img)
    plt.axis("off")

    for k in range(len(chans[keep_idx])):
        c = chans[keep_idx][k]
        
        # Plot single channel input
        plt.subplot(h, w, (3*k+3)*w+s+1)
        img =  x_test_i[np.newaxis, s, :, :, k]
        img = np.squeeze(img)
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        
        # Plot single channel ground truth
        plt.subplot(h, w, (3*k+4)*w+s+1)
        img = y_test_i[np.newaxis, s, :, :, k]
        img = np.squeeze(img)
        plt.imshow(img, cmap='gray')
        plt.axis("off")

        # Plot single channel output
        plt.subplot(h, w, (3*k+5)*w+s+1)
        img = _P_mean[:, :, :, k]
        img = np.squeeze(img)
        plt.imshow(img, cmap='gray')
        plt.axis("off")

fig.savefig('/home/ubuntu/candace/plots/epoch_50/' + ch[0] + '_' + ch[1] + '_' + str(nsamp) + 'samps.pdf')

