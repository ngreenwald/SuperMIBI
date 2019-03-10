# Script for visualizing the input, output, ground truth, and associated loss

import matplotlib.pyplot as plt
import numpy

# Directory to examples and accuracy values

dir = ''
X = numpy.load(dir + 'xs.npy')
Y = numpy.load(dir + 'ys.npy')
Yhat = numpy.load(dir + 'y_hat.npy')

# Loop through all examples and produce + save subplot

for i in range(0, X.shape[0]):

    X1 = X[i, :, :, 0]
    X2 = X[i, :, :, 1]

    Y1 = Y[i, :, :, 0]
    Y2 = Y[i, :, :, 1]

    Yhat1 = Yhat[i, :, :, 0]
    Yhat2 = Yhat[i, :, :, 1]

    accuracy = open((dir + 'metrics' + str(i)), 'r').read()

    fig = plt.figure()
    fig.suptitle(('Example ' + str(i) + ', ' + accuracy))

    # Creates a 2x3 subplot ie. one channel per row

    ax1 = fig.add_subplot(231)
    ax1.imshow(X1)
    ax1.title.set_text('Input')

    ax2 = fig.add_subplot(232)
    ax2.imshow(Y1)
    ax2.title.set_text('Ground Truth')

    ax3 = fig.add_subplot(233)
    ax3.imshow(Yhat1)
    ax3.title.set_text('Output')

    ax4 = fig.add_subplot(234)
    ax4.imshow(X2)
    ax4.title.set_text('Input')

    ax5 = fig.add_subplot(235)
    ax5.imshow(Y2)
    ax5.title.set_text('Ground Truth')

    ax6 = fig.add_subplot(236)
    ax6.imshow(Yhat2)
    ax6.title.set_text('Output')

    # Save the figure as a pdf for each example

    fig.savefig('Example' + str(i) + '.pdf')



