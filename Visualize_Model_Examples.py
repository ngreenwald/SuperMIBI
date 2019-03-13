# Script for visualizing the input, output, ground truth, and associated loss

import matplotlib.pyplot as plt
import numpy
import sys

# check command line arguments

if len(sys.argv) == 1:
    raise ValueError("Command line arguments not supplied")

else:
    model_name = sys.argv[1]

# Directory to examples and accuracy values

dir = 'output/metrics/' + model_name + '/'
X = numpy.load(dir + 'xs.npy')
Y = numpy.load(dir + 'ys.npy')
Yhat = numpy.load(dir + 'y_hat.npy')

# Loop through all examples and produce + save subplot

for i in range(0, X.shape[0]):
    if X.shape[3] == 1:  
        X1 = X[i, :, :, 0]
        X2 = X[i, :, :, 0]

        Y1 = Y[i, :, :, 0]
        Y2 = Y[i, :, :, 0]

        Yhat1 = Yhat[i, :, :, 0]
        Yhat2 = Yhat[i, :, :, 0]
    
    else:
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
    im1 = ax1.imshow(X1)
    plt.colorbar(im1)
    ax1.title.set_text('Input')

    ax2 = fig.add_subplot(232)
    im2 = ax2.imshow(Y1)
    plt.colorbar(im2)
    ax2.title.set_text('Ground Truth')

    ax3 = fig.add_subplot(233)
    im3 = ax3.imshow(Yhat1)
    plt.colorbar(im3)
    ax3.title.set_text('Output')

    ax4 = fig.add_subplot(234)
    im4 = ax4.imshow(X2)
    plt.colorbar(im4)
    ax4.title.set_text('Input')

    ax5 = fig.add_subplot(235)
    im5 = ax5.imshow(Y2)
    plt.colorbar(im5)
    ax5.title.set_text('Ground Truth')

    ax6 = fig.add_subplot(236)
    im6 = ax6.imshow(Yhat2)
    plt.colorbar(im6)
    ax6.title.set_text('Output')

    # Save the figure as a pdf for each example

    fig.savefig(dir + 'Example' + str(i) + '.pdf')

