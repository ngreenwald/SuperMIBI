import Models
import numpy as np
import h5py
base_dir = '/home/ubuntu/SuperMIBI/data/test_run/'

from tensorflow.keras import models

current_model = models.load_model('my_first_model')

x_data_test = h5py.File(base_dir + 'x_cropped_test.h5', 'r')
y_data_test = h5py.File(base_dir + 'y_cropped_test.h5','r') 

x_test = x_data_test['x_test'][:]
y_test = y_data_test['y_test'][:]

test_idx = np.arange(5)
y_hat = current_model.predict(x_test[test_idx, :, :, 5:7])
xs = x_test[test_idx, :, :, 5:7]
ys = y_test[test_idx, :, :, 5:7]

np.save('xs.npy', xs)
np.save('ys.npy', ys)
np.save('y_hat.npy', y_hat)
for i in test_idx:
    y_hat_metrics = current_model.evaluate(x_test[i:(i + 1), :, :, 5:7], y_test[i:(i + 1), :, :, 5:7])
    np.savetxt('metrics{}'.format(i), y_hat_metrics)
