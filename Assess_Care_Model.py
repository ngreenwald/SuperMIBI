from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import h5py

from csbdeep.utils import download_and_extract_zip_file, axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import sys


## TODO: update this script, which is copied from Care_Model.py, to do visualization only
if len(sys.argv) < 4:
    raise ValueError("did not include enough command line arguments. Syntax: python Assess_Care_Model.py  model_name num_epochs channel_name1.tif...")
else:
    model_name = sys.argv[1]
    num_epochs = int(sys.argv[2])
    input_channels = sys.argv[3:len(sys.argv)]

print("using {} as model name".format(model_name))
print("training for {} epochs".format(num_epochs))
print("using {} as channels".format(input_channels))
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

# make subset data to only analyze specific channels: will use the channel names supplied at command line
print('using previously saved channel names for subsetting')
chans = np.load(base_dir + 'chan_names.npy')

keepers = input_channels
keep_idx = np.isin(chans, keepers)
if np.sum(keep_idx) == 0:
    raise ValueError("Did not supply valid channel name")

print('analyzing the following channels: {}'.format(chans[keep_idx]))


x_train, x_test = x_train[:, :, :, keep_idx], x_test[:, :, :, keep_idx]
y_train, y_test = y_train[:, :, :, keep_idx], y_test[:, :, :, keep_idx]


# this code taken directly from FAQ, uses internal functions to do plotting
fig = plt.figure(figsize=(30,30))
_P = model.keras_model.predict(x_test[:5, :, :, :])
_P_mean  = _P[...,:(_P.shape[-1]//2)]
_P_scale = _P[...,(_P.shape[-1]//2):]
plot_some(x_test[:5, :, :, 0],y_test[:5, :, :, :],_P_mean,_P_scale,pmax=99.5)
fig.suptitle('5 example validation patches\n'      
             'first row: input (source),  '        
             'second row: target (ground truth),  '
             'third row: predicted Laplace mean,  '
             'forth row: predicted Laplace scale');
fig.savefig('/models/' + model_name + '.pdf')

