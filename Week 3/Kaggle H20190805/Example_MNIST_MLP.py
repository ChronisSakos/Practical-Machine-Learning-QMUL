#!/usr/bin/python
"""
 Example script for use with MNIST data; train the weights of a MLP
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
training_epochs = 2000
DropOutKeepProb = 0.8
learning_rate = 0.001
displayStep = 20

# Network architecture parameters
n_input    = 28*28   # The NMIST images consists of 784 pixels. These are the input feature space.
n_classes  = 10       # Regression output is single valued
n_hidden_1 = 28*28   # 1st layer num features

x = tf.placeholder(tf.float32, shape=[None, 28*28], name="x")
y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")


print("Creating a layer with ", n_hidden_1, " nodes")
w_layer_1    = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]))
layer_1      = tf.nn.relu(tf.add(tf.matmul(x,w_layer_1),bias_layer_1))


# Similarly we now construct the output of the network, where the output layer
# combines the information down into a space of evidences for the possible
# classes in the problem (n_classes=1 for this regression problem).
print("Creating the output layer ", n_classes, " output values")
output       = tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
bias_output  = tf.Variable(tf.random_normal([n_classes]))
model  = tf.matmul(layer_1, output) + bias_output
dmodel = tf.nn.dropout(model, keep_prob)


#
# Set parameters for training; cross entropy loss function, with the Adam optimiser
#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=dmodel))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
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
# run the MLP training phase, using batches of 50 images for each train sample.  From these
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
  train_step.run(feed_dict={x: batch_img, y: batch_lbl, keep_prob: DropOutKeepProb})

# now compute the accuracy for train/test samples as a funciton of epoch for every
# nth epoch.  For this stage ensure that the keep prob is set to 1.0 to evaluate the 
# performance of the network including all nodes (as it will ultimately be used).
  if not i % displayStep:
    train_accuracy = accuracy.eval(feed_dict={x:batch_img, y: batch_lbl, keep_prob: 1.0})
    test_accuracy  = accuracy.eval(feed_dict={x: testbatch[0], y: testbatch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g   test accuracy %g"%(i, train_accuracy, test_accuracy))

    cost_test.append(  sess.run(cross_entropy, feed_dict={x: batch_img, y: batch_lbl, keep_prob: 1.0}) )
    cost_train.append( sess.run(cross_entropy, feed_dict={x: testbatch[0], y: testbatch[1], keep_prob: 1.0}) )

    accuracy_train.append(train_accuracy)
    accuracy_test.append(test_accuracy)
    epoch.append(i+1)

#    print("model predictions")
#    print(sess.run(model, feed_dict={x: batch_img}))
#    print("label predictions")
#    print(batch_lbl)

print ("Training phase finished")
print (" accuracy for the train set examples      = " , accuracy_train[-1])
print (" accuracy for the test set examples       = " , accuracy_test[-1])

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


