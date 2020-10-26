"""
This script is part of the Week 2 assignments
We adapted the Examples_LinearRegression.py script in order to evaluate the 
fit convergence behaviour for the different scenarios of gradient and 
intercept values.

------------------------------------------------------------------------
  author:       Chronis Sakos (c.sakos@stu18.qmul.ac.uk)
  Student ID:   190744335
  Date:         02/08/2019
  ----------------------------------------------------------------------
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Network training parameters
learning_rate    = 0.005
training_epochs  = 100
min_x = 0
max_x = 1
Ngen  = 100
gradient  = 1.0        # m values [1,10,100]
intercept = 0.0         # c
noise     = 0.1  # data are inherently noisy, so we generate noise

print("--------------------------------------------")
print("Number of examples to generate = ", Ngen)
print("Learning rate            alpha = ", learning_rate)
print("Number of training epochs      = ", training_epochs)
print("--------------------------------------------")


# tf Graph input:
#  x_: is the tensor for the input data (the placeholder entry None is used for that;
#     and the number of features input (n_input = 1).
#
#  y_: is the tensor input data computed from the generated data x values
#
#
#  y: is the tensor for the output value of the function that is being approximated by 
#     the the model, calculated from y = mx+c
x_ = tf.placeholder(tf.float32, [None, 1], name="x_")
y_ = tf.placeholder(tf.float32, [None, 1], name="y_")

# parameters of the model are m (gradient) and c (constant offset)
#   - pick random starting values for fit convergence
c = tf.Variable(tf.random_uniform([1]), name="c")
m = tf.Variable(tf.random_uniform([1]), name="m")
y = m * x_ + c

# assume all data have equal uncertainties to avoid having to define an example by
# example uncertainty, and define the loss function as a simplified chi^2 sum
loss = tf.reduce_sum((y - y_) * (y - y_))

# use a gradient descent optimiser
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# prepare data - data_x and data_y are np arrays that will get fed into the optimisation
# set the seed for random number generation to ensure we always get the same data
np.random.seed(0)
data_x = (max_x-min_x)*np.random.random([Ngen, 1])+min_x
data_y = np.zeros([Ngen,1])
for i in range(Ngen):
    data_y[i] = gradient*data_x[i]+intercept + noise*np.random.randn()


# prepare the session for evaluation and initialise the variables.
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# epoch and chi2 are the training step and the chi2 sum (neglecting the error normalisation)
# for the problem.  These will be logged to inspect the fit convergence obtained.
epoch = []
chi2  = []
the_c  = []
the_m  = []
print("Epoch   m         c")
for step in range(training_epochs):
    # run the minimiser
    sess.run(train_step, feed_dict={x_: data_x, y_: data_y})

    # log the data and print output
    epoch.append(step)
    chi2.append(sess.run(loss, feed_dict={x_: data_x, y_: data_y}))
    the_c.append(sess.run(c))
    the_m.append(sess.run(m))
    # uncomment this line for verbose printout
    #print(step, sess.run(m), sess.run(c), chi2[step])


# print optimisation results
print("Optimisation process has finished.  The optimal values of parameters are:")
print("  m = ", sess.run(m), " [input value = ", gradient, "]" )
print("  c = ", sess.run(c), " [input value = ", intercept, "]" )
print("  loss function for the optimal parameters = ", chi2[training_epochs-1])

plt.plot(data_x, data_y, 'r.')
plt.plot([min_x, max_x], [sess.run(c), sess.run(m)*max_x+sess.run(c)], 'b-')
plt.ylabel('f(x)')
plt.xlabel('x')
plt.title('Linear Regression Example')
plt.show()

plt.plot(epoch, chi2)
plt.ylabel('Loss')
plt.xlabel('Training epoch')
plt.title('Linear Regression Example')
plt.show()

plt.plot(epoch, the_c)
plt.ylabel('c')
plt.xlabel('Training epoch')
plt.title('Linear Regression Example')
plt.show()

plt.plot(epoch, the_m)
plt.ylabel('m')
plt.xlabel('Training epoch')
plt.title('Linear Regression Example')
plt.show()
