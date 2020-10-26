#!/usr/bin/python
"""
Example_Overfitting.py

Construct a single layer regression MLP to use as a function approximator.
Deliberately feed in a small data sample to illustrate how models can 
overfit for regions of the input feature space.

This python script has been written for use with the Statistical Data Analysis
course used for the Queen Mary University of London .

  ----------------------------------------------------------------------
  author:       Adrian Bevan (a.j.bevan@qmul.ac.uk)
  Copyright (C) QMUL 2019
  ----------------------------------------------------------------------
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# this is required to permit multiple copies of the OpenMP runtime to be linked
# to the programme.  Failure to include the following two lines will result in
# an error that Spyder will not report.  On PyCharm the error provided will be
#    OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
#    ...
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Network training parameters
learning_rate    = 0.005
training_epochs  = 2500
min_x = -10
max_x = 10
Ngen  = 10
gradient = tf.constant(1.0)
offset   = tf.constant(0.0)


# Network architecture parameters
n_input    = 1   # 1D function, so there is one input feature per example
n_classes  = 1   # Regression output is single valued
n_hidden_1 = 100  # 1st layer num features

print("--------------------------------------------")
print("Number of input features       = ", n_input)
print("Number of output classes       = ", n_classes)
print("Number of examples to generate = ", Ngen)
print("Learning rate            alpha = ", learning_rate)
print("Number of training epochs      = ", training_epochs)
print("--------------------------------------------")

def myFunctionTF(arg):
    """
    User defined function for the MLP to learn.  The default example is 
    the square root function.
    """
    return gradient*arg*arg+offset+tf.random_normal([1], 0.0, 1.0)


# tf Graph input:
#  x: is the tensor for the input data (the placeholder entry None is used for that;
#     and the number of features input (n_input = 1).
#
#  y: is the tensor for the output value of the function that is being approximated by 
#     the MLP.
#
x = tf.placeholder(tf.float32, [None, n_input], name="x")
y = tf.placeholder(tf.float32, [None, n_classes], name="y")


# We construct layer 1 from a weight set, a bias set and the activiation function used
# to process the impulse set of features for a given example in order to produce a 
# predictive output for that example.
#
#  w_layer_1:    the weights for layer 1.  The first index is the input feature (pixel)
#                and the second index is the node index for the perceptron in the first
#                layer.
#  bias_layer_1: the biases for layer 1.  There is a single bias for each node in the 
#                layer.
#  layer_1:      the activation functions for layer 1
#
print("Creating a hidden layer with ", n_hidden_1, " nodes")
w_layer_1    = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]))
layer_1      = tf.nn.sigmoid(tf.add(tf.matmul(x,w_layer_1),bias_layer_1))


# Similarly we now construct the output of the network, where the output layer
# combines the information down into a space of evidences for the possible
# classes in the problem (n_classes=1 for this regression problem).
print("Creating the output layer ", n_classes, " output values")
output       = tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
bias_output  = tf.Variable(tf.random_normal([n_classes]))
output_layer = tf.matmul(layer_1, output) + bias_output

#optimise with l2 loss function
print("Using the L2 loss function implemented in tf.nn")
loss = tf.nn.l2_loss(y - output_layer)

# optimizer: take the Adam optimiser, see https://arxiv.org/pdf/1412.6980v8.pdf for
# details of this algorithm.
print("Using the Adam optimiser to train the network")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# generate data, the input data is a random number betwen 1 and 10,
# and the corresponding label value is the square root of that number
print("Generating the test and training sets.  There are ", Ngen, " examples in each")
tftraindata = tf.random_uniform([Ngen, 1], min_x, max_x)  # training set
tftestdata  = tf.random_uniform([Ngen, 1], min_x, max_x)  # test set

# define operation for computing the regression output
probabilities = output_layer

# Initializing the variables
init  = tf.global_variables_initializer()

# Start the session to embark on the training cycle
sess = tf.Session()
sess.run(init)

# convert the training data to np arrays so that these can be used with the feed_dict when training
traindata  = sess.run(tftraindata) 
target_value = sess.run(myFunctionTF(traindata))

# convert the test data to np arrays so that these can be used with the feed_dict when training
testdata  = sess.run(tftestdata) 
test_value = sess.run(myFunctionTF(testdata))

# plot the test and training data
plotting = 0
if plotting:
    # plot the function f(x)=mx+c
    plt.subplot(2, 1, 1)
    plt.plot(testdata, test_value, 'r.')
    plt.ylabel('f(x) = sqr(x)')
    plt.xlabel('x')
    plt.title('Training data')
    plt.subplot(2, 1, 2)
    plt.plot(traindata, target_value, 'b.')
    plt.ylabel('f(x) = sqr(x)')
    plt.xlabel('x')
    plt.title('Test data')
    plt.show()

# arrays to compare the input value vs the prediction
input_value      = []
epoch_set        = []
loss_test_value  = []
loss_set         = []
prediction_value = []
true_value       = []


# Training cycle - for this example we do not batch train as this is a shallow network with 
# a low number of training events.
for epoch in range(training_epochs):
    the_loss = 0.
    
    print("Training epoch number ", epoch)
    sess.run(optimizer, feed_dict={x: traindata, y: target_value})
    the_loss = sess.run(loss, feed_dict={x: traindata, y: target_value})

    loss_set.append(the_loss)
    epoch_set.append(epoch+1)
    
    the_loss = sess.run(loss, feed_dict={x: testdata, y: test_value})
    loss_test_value.append(the_loss)

    #
    # This is a regression analysis problem, so we want to evaluate and display the output_layer
    # value (model response function), and not an output prediction (which would have been appropraite
    # for a classification problem)
    #
    if epoch == training_epochs-1:
        step = (max_x - min_x)/100
        for i in range(100):
            thisx = min_x + i*step
            pred = probabilities.eval(feed_dict={x: [[thisx]]}, session=sess)
            print ("x = ", thisx, ", prediction =", pred)
            input_value.append(thisx)
            prediction_value.append(pred[0])
            true_value.append(gradient.eval(session=sess)*thisx*thisx+offset.eval(session=sess))

            pred = probabilities.eval(feed_dict={x: [[-thisx]]}, session=sess)
            print ("x = ", -thisx, ", prediction =", pred)
            input_value.append(-thisx)
            prediction_value.append(pred[0])

            true_value.append(gradient.eval(session=sess)*thisx*thisx+offset.eval(session=sess))


# Result summary
print ("Training phase finished")

#plt.subplot(2, 1, 1)
plt.plot(testdata, test_value, 'bo')
plt.ylabel('f(x) = sqr(x)')
plt.xlabel('x')
plt.title('Test data')
plt.plot(input_value, prediction_value, 'r*')
plt.plot(input_value, true_value, 'x')
plt.ylabel('f(x) = sqr(x)')
plt.xlabel('x')
plt.title('Network Response Function')

plt.show()
