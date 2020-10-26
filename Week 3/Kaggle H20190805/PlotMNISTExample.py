#!/usr/bin/python
"""

 Example script for use with MNIST data; Select the specified training example
 to plot and show this image.

This python script has been written for use with the PracticalMachineLearning
course used for the Queen Mary University of London Summer School.

Please see
   https://qmplus.qmul.ac.uk/course/view.php?id=10006
  ----------------------------------------------------------------------
  author:       Adrian Bevan (a.j.bevan@qmul.ac.uk)
  Copyright (C) QMUL 2019
  ----------------------------------------------------------------------
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt

print("Importing the MNIST data via input_data interface")
mnist = read_data_sets("/tmp/data/", one_hot=True)

example = 4
print("Importing the MNIST data")
mnist = read_data_sets("/tmp/data/", one_hot=True)

# the MNIST image data are arrays of 28x28 pixels, and the set
# has 10 target classification types (the numbers zero through
# nine).
img_size_x = 28
img_size_y = 28
img_size = img_size_x*img_size_y

print("Plotting example MNIST image number ", example)
print("Running an interactive session")
sess = tf.InteractiveSession()

# get the mnist data and reshape the array into a 28*28 image
data = mnist.train.images[example:example+1,:]
label = mnist.train.labels[example:example+1,:]
image = data[0].reshape([28,28])

# plot the image, and note the example number and label of the image
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.title('Example: %d  Label: %d' % (example, label[0].argmax(axis=0)))
plt.show()
