import Models
import numpy as np
import h5py
import sys
base_dir = '/home/ubuntu/SuperMIBI/data/test_run/'
from tensorflow.keras import models
import os
import random

# check for command line args

if len(sys.argv) == 1:
    raise ValueError("Did not supply command line arguments")
else:
    model_name = sys.argv[1]

current_model = models.load_model('output/models/' + model_name)

x_data_test = h5py.File(base_dir + 'x_cropped_test.h5', 'r')
y_data_test = h5py.File(base_dir + 'y_cropped_test.h5','r') 

# load channels that were used to train the model
keep_idx = np.load('output/models/' + model_name + '_channel_idx.npy')

x_test = x_data_test['x_test'][:]
y_test = y_data_test['y_test'][:]

test_idx = random.sample(range(x_test.shape[0]), 15)
xs = x_test[test_idx, :, :, :]
xs = xs[:, :, :, keep_idx]
ys = y_test[test_idx, :, :, :]
ys = ys[:, :, :, keep_idx]

y_hat = current_model.predict(xs)
out_dir = 'output/metrics/' + model_name + '/'

if os.path.isdir(out_dir):
    raise ValueError("directory already exists")

os.mkdir(out_dir)
np.save(out_dir + 'xs.npy', xs)
np.save(out_dir + 'ys.npy', ys)
np.save(out_dir + 'y_hat.npy', y_hat)
for i in range(len(test_idx)):
    y_hat_metrics = current_model.evaluate(xs[i:(i + 1), :, :, :], ys[i:(i + 1), :, :, :])
    np.savetxt(out_dir + 'metrics{}'.format(i), y_hat_metrics)
