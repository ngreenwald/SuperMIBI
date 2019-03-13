import matplotlib.pyplot as plt
import numpy as np

wd = '/Users/noahgreenwald/Documents/Grad_School/Classes/CS230/'

x_data = np.load(wd + 'output/xs.npy')
y_data = np.load(wd + 'output/ys.npy')

x_data.shape

plt.imshow(x_data[0, :, :, 0])
plt.imshow(y_data[0, :, :, 0])