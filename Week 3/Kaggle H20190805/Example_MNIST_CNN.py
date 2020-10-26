#!/usr/bin/python
"""
 Example script for use with MNIST data; train the weights of a CNN
 using the gradient descent optimiser.  The cost function used for 
 this is cross entropy.  This example is based on:

   https://www.tensorflow.org/get_started/mnist/pros

This python script has been written for use with the PracticalMachineLearning
course used for the Queen Mary University of London Summer School.

Please see
   https://qmplus.qmul.ac.uk/course/view.php?id=10006
  ----------------------------------------------------------------------
  author:       Adrian Bevan (a.j.bevan@qmul.ac.uk)
  Copyright (C) QMUL 2019
  ----------------------------------------------------------------------
"""
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt

# this is required to permit multiple copies of the OpenMP runtime to be linked
# to the programme.  Failure to include the following two lines will result in
# an error that Spyder will not report.  On PyCharm the error provided will be
#    OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
#    ...
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# read in the data and start an interactive session
print ("Importing the MNIST data")
mnist = read_data_sets("./", one_hot=True)

# this problem is classifying handwriting, so there are 10 possible outputs:
#    0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
# and each image is represented by 784 features (28x28) that can be processed
# by the CNN
n_outputs= 10
image_x  = 28
image_y  = 28
training_epochs = 100
DropOutKeepProb = 0.5
learning_rate = 0.001
displayStep = 20
# the image shape is dictated by the input data.  This example uses MNIST data,
# arranged as a 28 x 28 array of pixels
imageShape = [-1, image_x, image_y, 1]

x_ = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  # function to create weight variables with the specified shape that are randomly initialised.
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  # function to create bias variables with the specified shape that are randomly initialised.
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
# simple interface to the nn.conv2d function
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
# simple interface to the nn.max_pool function
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x_, [-1,28,28,1])

# layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = conv2d(x_image, W_conv1) + b_conv1
h_pool1 = max_pool_2x2(h_conv1)

# layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
h_pool2 = max_pool_2x2(h_conv2)


# fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

model = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#
# Set parameters for training; cross entropy loss function, with the Adam optimiser
#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=model))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(model,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#
# Run the training over N epochs
#
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

accuracy_train=[]
accuracy_test=[]
cost_test=[]
cost_train=[]
epoch=[]
# run the CNN training phase, using batches of 50 images for each train sample.  From these
# extract the accuracy of predictions and cost in order to show the performance of the CNN 
# as a function of epoch for the test and train sample.
#
# Also compute the final performance against the validation sample.
for i in range(training_epochs):
  # we can either split data and lables into two variables for the examples, or 
  # we can combine them and refer to the data examples as X[0] and the labels as
  # X[1] as shown for the train and test examples in the following.
  batch_img, batch_lbl = mnist.train.next_batch(50)
  testbatch = mnist.test.next_batch(50)

# train the hyper parameters for the CNN; use a drop out keep prob of DropOutKeepProb
# to promote generalisation of the network.
  train_step.run(feed_dict={x_: batch_img, y_: batch_lbl, keep_prob: DropOutKeepProb})

# now compute the accuracy for train/test samples as a funciton of epoch for every
# nth epoch.  For this stage ensure that the keep prob is set to 1.0 to evaluate the 
# performance of the network including all nodes (as it will ultimately be used).
  if not i % displayStep:
    train_accuracy = accuracy.eval(feed_dict={x_:batch_img, y_: batch_lbl, keep_prob: 1.0})
    test_accuracy =accuracy.eval(feed_dict={x_: testbatch[0], y_: testbatch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g   test accuracy %g"%(i, train_accuracy, test_accuracy))

    cost_test.append( sess.run(cross_entropy, feed_dict={x_: batch_img, y_: batch_lbl, keep_prob: 1.0}) )
    cost_train.append( sess.run(cross_entropy, feed_dict={x_: testbatch[0], y_: testbatch[1], keep_prob: 1.0}) )

    accuracy_train.append(train_accuracy)
    accuracy_test.append(test_accuracy)
    epoch.append(i+1)

# now that the training has finished we should compute the accuracy of the final network on test, train 
# and also the validation samples.  This is important in order to get comparative benchmarks and demonstrate
# that the performance is repeatable on different samples of events, including those previously unseen
# events.
print("Model has been trained - will now evaluate accuracy")
train_accuracy = accuracy.eval(feed_dict={x_: mnist.test.images[0:2000], y_: mnist.test.labels[0:2000], keep_prob: 1.0})
test_accuracy  = accuracy.eval(feed_dict={x_: mnist.train.images[0:2000], y_: mnist.train.labels[0:2000], keep_prob: 1.0})
val_accuracy   = accuracy.eval(feed_dict={x_: mnist.validation.images[0:2000], y_: mnist.validation.labels[0:2000], keep_prob: 1.0})

print ("Training phase finished")
print (" accuracy for the train set examples      = " , train_accuracy)
print (" accuracy for the test set examples       = " , test_accuracy)
print (" accuracy for the validation set examples = " , val_accuracy)

plt.figure(1)
plt.subplot(211)
plt.plot(epoch, accuracy_train, 'o', label='Logistic regression training phase')
plt.plot(epoch, accuracy_test, 'o', label='Logistic regression testing phase')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc=4)

plt.subplot(212)
plt.plot(epoch, cost_train, 'o', label='Logistic regression training phase')
plt.plot(epoch, cost_test, 'o', label='Logistic regression testing phase')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend()
plt.show()


