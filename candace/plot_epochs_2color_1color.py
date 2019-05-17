# Plot results from different epochs on Ki67 and CD45 channels
# Plot both channels together and each channel individually
# Plot 1 sample per pdf

from csbdeep.models import CARE
from csbdeep.utils import plot_some, to_color
import h5py
import numpy as np
import re
import matplotlib.pyplot as plt

names = ['epoch5_ki67_cd45', 'epoch10_ki67_cd45', 'epoch50_ki67_cd45', 'epoch100_ki67_cd45']
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

for s in range(nsamp):

    fig = plt.figure(figsize=(30,30))
    plt.gcf().text(0.05, 0.82, "Input", fontsize=24)
    plt.gcf().text(0.05, 0.688, "Ground truth", fontsize=24)
    plt.gcf().text(0.05, 0.556, "Epoch 5", fontsize=24)
    plt.gcf().text(0.05, 0.424, "Epoch 10", fontsize=24)
    plt.gcf().text(0.05, 0.292, "Epoch 50", fontsize=24)
    plt.gcf().text(0.05, 0.16, "Epoch 100", fontsize=24)

    # Assumes same channels for all the models
    n = names[0]
    ch = [re.search('_(.+)_',n).group(1), re.search('_.+_(.+)$',n).group(1)]
    keepers = [s + '.tif' for s in ch]
    keepers_lower = [x.lower() for x in keepers]
    keep_idx = np.isin(chans_lower, keepers_lower)
    print('analyzing the following channels: {}'.format(chans[keep_idx]))
    x_test_i = x_test[:, :, :, keep_idx]
    y_test_i = y_test[:, :, :, keep_idx]

    h = len(names)+2 #1 row for input, 1 row for ground truth, 1 row for each epoch
    w = 3 #1 column for both channels, 1 column for each channel individually

    # Plot input
    plt.subplot(h, w, 1)
    plt.title(chans[keep_idx][0] + ", " + chans[keep_idx][1], fontsize=24)
    img = x_test_i[np.newaxis, s, :, :, :]
    img = np.squeeze(np.stack(map(to_color,img)))
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(h, w, 2)
    plt.title(chans[keep_idx][0], fontsize=24)
    img = x_test_i[np.newaxis, s, :, :, 0]
    img = np.squeeze(img)
    plt.imshow(img, cmap='gray')
    plt.axis("off")

    plt.subplot(h, w, 3)
    plt.title(chans[keep_idx][1], fontsize=24)
    img = x_test_i[np.newaxis, s, :, :, 1]
    img = np.squeeze(img)
    plt.imshow(img, cmap='gray')
    plt.axis("off")

    # Plot ground truth
    plt.subplot(h, w, 4)
    img = y_test_i[np.newaxis, s, :, :, :]
    img = np.squeeze(np.stack(map(to_color,img)))
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(h, w, 5)
    img = y_test_i[np.newaxis, s, :, :, 0]
    img = np.squeeze(img)
    plt.imshow(img, cmap='gray')
    plt.axis("off")

    plt.subplot(h, w, 6)
    img = y_test_i[np.newaxis, s, :, :, 1]
    img = np.squeeze(img)
    plt.imshow(img, cmap='gray')
    plt.axis("off")

    for i in range(len(names)):
        n = names[i]
        epoch = re.search('^[^_]+(?=_)',n).group(0)

        print('analyzing the model: '+n) 
        model = CARE(config=None, name=n, basedir=models_basedir)

        _P = model.keras_model.predict(x_test_i[np.newaxis, s, :, :, :])
        _P_mean  = _P[...,:(_P.shape[-1]//2)]

        plt.subplot(h, w, (i+2)*w + 1)
        img = _P_mean
        img = np.squeeze(np.stack(map(to_color,img)))
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(h, w, (i+2)*w + 2)
        img = _P_mean[:,:,:,0]
        img = np.squeeze(img)
        plt.imshow(img, cmap='gray')
        plt.axis("off")

        plt.subplot(h, w, (i+2)*w + 3)
        img = _P_mean[:,:,:,1]
        img = np.squeeze(img)
        plt.imshow(img, cmap='gray')
        plt.axis("off")

    fig.savefig('/home/ubuntu/candace/plots/ki67_cd45_epochs/' + ch[0] + '_' + ch[1] + '_singlechan_' + str(s) + '.pdf')

