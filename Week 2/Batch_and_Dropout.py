"""
This script is part of the Week 2 assignments
We adapted the Example_FunctionApproximator.py script in order to 
effect of batch training to the MLP.
We use the MLP with the 2 hidden layers that we created before.
We divide the dataset into 30 groups/batches and make use of Dropout 
in the 2nd hidden layer, with keep probability of 50%. 
The training epoch for each one of them is 100

------------------------------------------------------------------------
  author:       Chronis Sakos (c.sakos@stu18.qmul.ac.uk)
  Student ID:   190744335
  Date:         03/08/2019
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
learning_rate    = 0.01
training_epochs  = 100
min_x = -10
max_x = 10
Ngen  = 10000

keep_prob = 0.5     #keep probability - only used with the dropout functionality

batch_number = 30
batch_size = int(Ngen/batch_number)

# Network architecture parameters
n_input    = 1   # 1D function, so there is one input feature per example
n_classes  = 1   # Regression output is single valued
n_hidden_1 = 50  # 1st layer num features
n_hidden_2 = 50 # 2nd layer num features

#Flags for plotting and printing out.  If plotting is set to True then in addition to the 
# output plots, the script will make plots of the test and train data.  The verbose printout
# option set to True will result in detailed output being printed.
plotting = True
verbose = True

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
    return tf.square(arg)


# tf Graph input:
#  x_: is the tensor for the input data (the placeholder entry None is used for that;
#     and the number of features input (n_input = 1).
#
#  y_: is the tensor for the output value of the function that is being approximated by 
#     the MLP.
#
x_ = tf.placeholder(tf.float32, [None, n_input], name="x_")
y_ = tf.placeholder(tf.float32, [None, n_classes], name="y_")


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
print("Creating a hidden layer with ", n_hidden_1, " nodes")    #1st hidden layer
w_layer_1    = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]))
layer_1      = tf.nn.relu(tf.add(tf.matmul(x_,w_layer_1),bias_layer_1))


print("Creating a second hidden layer with ", n_hidden_2, " nodes")     #2nd hidden layer
w_layer_2    = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
bias_layer_2 = tf.Variable(tf.random_normal([n_hidden_2]))
layer_2      = tf.nn.relu(tf.add(tf.matmul(layer_1,w_layer_2),bias_layer_2))
dmodel_2     = tf.nn.dropout(layer_2, keep_prob) 


print("Creating the output layer ", n_classes, " output values")    #output layer
output       = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
bias_output  = tf.Variable(tf.random_normal([n_classes]))
# define operation for computing the regression output - this is our model prediction
probabilities = tf.matmul(dmodel_2, output) + bias_output

#optimise with l2 loss function
print("Using the L2 loss function implemented in tf.nn")
loss = tf.nn.l2_loss(y_ - probabilities)
# Alternative way to write the loss function
#loss = tf.reduce_sum((y_ - probabilities)*(y_ - probabilities))

# optimizer: take the Adam optimiser, see https://arxiv.org/pdf/1412.6980v8.pdf for
# details of this algorithm.
print("Using the Adam optimiser to train the network")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# arrays to compare the input value vs the prediction
input_value      = []
epoch_set        = []
loss_test_value  = []
loss_set         = []
prediction_value = []

#loop for every batch to run on its own and train its own data for specific training epochs 
j = 0
for j in range(batch_number):
    print('Batch number: ', j+1)
    tftraindata = tf.random_uniform([batch_size, 1], min_x, max_x)  # training set
    tftestdata  = tf.random_uniform([batch_size, 1], min_x, max_x)  # test set
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
    for epoch in range(training_epochs):
        the_loss = 0.
        if verbose:
            print("Training epoch number ", epoch)
            sess.run(optimizer, feed_dict={x_: traindata, y_: target_value})
            the_loss = sess.run(loss, feed_dict={x_: traindata, y_: target_value})
            
            loss_set.append(the_loss)
            epoch_set.append(epoch+1)
            
            the_loss = sess.run(loss, feed_dict={x_: testdata, y_: test_value})
            loss_test_value.append(the_loss)

    #
    # This is a regression analysis problem, so we want to evaluate and display the output_layer
    # value (model response function), and not an output prediction (which would have been appropraite
    # for a classification problem)
    #
step = (max_x - min_x)/100
for i in range(100):
    thisx = min_x + i*step
    pred = probabilities.eval(feed_dict={x_: [[thisx]]}, session=sess)
    if verbose:
        print ("x = ", thisx, ", prediction =", pred)
    input_value.append(thisx)
    prediction_value.append(pred[0])
    
    pred = probabilities.eval(feed_dict={x_: [[-thisx]]}, session=sess)
    if verbose:
        print ("x = ", -thisx, ", prediction =", pred)
    input_value.append(-thisx)
    prediction_value.append(pred[0])
            
# check the loss function for the last epoch vs the number of data
print("Loss function for the final epoch = ", loss_test_value[-1], " (test data)")
print("Pseudo chi^2 = loss/Ndata         = {0:4f} (test data)".format( loss_test_value[-1]/Ngen ))
# Result summary
print ("Training phase finished")

plt.subplot(2, 1, 1)
plt.plot(testdata, test_value, 'b.')
plt.ylabel('f(x) = sqr(x)')
plt.xlabel('x')
plt.title('Test data')
plt.plot(input_value, prediction_value, 'r*')
plt.ylabel('f(x) = sqr(x)')
plt.xlabel('x')
plt.title('Network Response Function')

ax = plt.subplot(2, 1, 2)
plt.plot(epoch_set, loss_set, 'o', label='MLP Training phase')
plt.plot(epoch_set, loss_test_value, 'rx', label='MLP Training phase')
plt.ylabel('loss')
ax.set_yscale('log')
plt.xlabel('epoch')

plt.show()
