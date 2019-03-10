# Script for visualizing the input, output, ground truth, and associated loss

import matplotlib.pyplot as plt

# Directory to tiff examples and loss value

tiff_dir = ''
loss_dir = ''
loss = ''

# Loop through all examples and produce + save subplot

for i in range(1, 6):

    X = plt.imread(tiff_dir+'X'+str(i)+'.tiff')
    Y = plt.imread(tiff_dir+'Y'+str(i)+'.tiff')
    Yhat = plt.imread(tiff_dir+'Yhat'+str(i)+'.tiff')

    fig = plt.figure()
    fig.suptitle(('Example ' + str(i) + ', ' + loss))

    ax1 = fig.add_subplot(131)
    ax1.imshow(X)
    ax1.title.set_text('Input')

    ax2 = fig.add_subplot(132)
    ax2.imshow(Y)
    ax2.title.set_text('Ground Truth')

    ax3 = fig.add_subplot(133)
    ax3.imshow(Yhat)
    ax3.title.set_text('Output')

    fig.savefig('Example ' + str(i) + '.pdf')



