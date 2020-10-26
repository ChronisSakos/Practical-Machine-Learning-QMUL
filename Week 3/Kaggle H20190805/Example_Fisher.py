#!/usr/bin/python
"""
 Compute the coefficients of a Fisher discriminant given some 
 data randomly generated according to a normal distrubtion for
 a 2D problem.

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
import Fisher as Fish
import matplotlib.pyplot as plt
import numpy as np

ndim = 2
N = 1000
muA = [1.0, 0.7]
muB = [-1.0, -1.0]
sigmaA = [0.6, 0.5]
sigmaB = [0.5, 0.4]
columns = ['x', 'y']

# use random_normal to generate the data
classA = tf.Variable(tf.random_normal([N, ndim], muA, sigmaA), name = "data_for_class_A")
classB = tf.Variable(tf.random_normal([N, ndim], muB, sigmaB), name = "data_for_class_A")

fish = Fish.Fisher("myfisher")
fish.CalcCoefficients(classA, classB, columns)
fish.Print()

# the fisher class instance has a session that we can use to convert the TF variable to a
# numpy array
dataClassA = fish.sess.run(classA)
dataClassB = fish.sess.run(classB)

cnstClassA = tf.constant(dataClassA)
cnstClassB = tf.constant(dataClassB)

FishDataA = fish.ProcessData(classA);
FishDataB = fish.ProcessData(classB);
FA = fish.sess.run(FishDataA)
FB = fish.sess.run(FishDataB)

# make a scatter plot of the generated data
plt.scatter(dataClassA[:, 0], dataClassA[:, 1], c='b', alpha=0.5, label='Class A')
plt.scatter(dataClassB[:, 0], dataClassB[:, 1], c='r', alpha=0.5, label='Class B')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('y')

# set appropriate plot ranges to avoid having a crowded image
plt.xlim(-3, 4)
plt.ylim(-3, 4)
plt.savefig("FisherExample_xy.pdf")
plt.show()

# make a scatter plot of the generated data
xybins = np.linspace(-3, 4, 100)
plt.clf()                   # clear the figure to start a new plot
plt.hist(dataClassA[:, 0], xybins, alpha=0.5, facecolor='b', label='Class A')
plt.hist(dataClassB[:, 0], xybins, alpha=0.5, facecolor='r', label='Class B')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('Number of examples')
plt.savefig("FisherExample_x.pdf")
plt.show()

# make a scatter plot of the generated data
xybins = np.linspace(-3, 4, 100)
plt.clf()                   # clear the figure to start a new plot
plt.hist(dataClassA[:, 1], xybins, alpha=0.5, facecolor='b', label='Class A')
plt.hist(dataClassB[:, 1], xybins, alpha=0.5, facecolor='r', label='Class B')
plt.legend(loc='upper left')
plt.xlabel('y')
plt.ylabel('Number of examples')
plt.savefig("FisherExample_y.pdf")
plt.show()


# make a plot of Fisher distributions for the different classes of data
bins = np.linspace(-15, 15, 100)
plt.clf()                   # clear the figure to start a new plot
plt.hist(FA, bins, alpha=0.5, facecolor='b', label='Class A')
plt.hist(FB, bins, alpha=0.5, facecolor='r', label='Class B')
plt.legend(loc='upper right')
plt.xlabel('F')
plt.ylabel('Number of examples')
plt.savefig("FisherOutput.pdf")


