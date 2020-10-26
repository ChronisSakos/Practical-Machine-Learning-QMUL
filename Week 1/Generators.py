"""
------------------------------------------------------------------------
  author:       Chronis Sakos (c.sakos@stu18.qmul.ac.uk)
  Student ID:   190744335
  Date:         28/07/2019
  ----------------------------------------------------------------------
"""

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# this is required to permit multiple copies of the OpenMP runtime to be linked
# to the programme.  Failure to include the following two lines will result in
# an error that Spyder will not report.  On PyCharm the error provided will be
#    OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
#    ...
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


Ngen = 10000
tdata_normal  = tf.random.normal([Ngen], 3, 1.0)  # arguments are [Shape], mean, width
tdata_exponential = tf.random.gamma([Ngen], 1)   # arguments are [Shape], aplpha
tdata_poisson = tf.random.poisson(4, [Ngen])    #argument are lamda, [Shape]

# Initializing the variables
init = tf.global_variables_initializer()
with  tf.Session() as sess:
    sess.run(init)

    data_normal  = sess.run(tdata_normal)
    data_exponential = sess.run(tdata_exponential)
    data_poisson = sess.run(tdata_poisson)

    """ Plotting the normal distribution """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    n, bins, patches = ax.hist(data_normal, 50, (-1, 7))
    ax.set_xlabel("x")
    ax.set_ylabel("Number of entries")
    fig.savefig("Chronis_Sakos_normal.pdf")
    plt.title("Normal Distribution")
    plt.show()
    plt.close(fig)    
    print("\tA normal distribution of events has been generated and the results printed in Chronis_Sakos_normal.pdf")

    """ Plotting the exponential distribution """
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    n, bins, patches = ax2.hist(data_exponential, 50, (0, 7))
    ax2.set_xlabel("x")
    ax2.set_ylabel("Number of entries")
    fig2.savefig("Chronis_Sakos_exp.pdf")
    plt.title("Exponential Distribution")
    plt.show()
    plt.close(fig2)    
    print("\tAn exponential distribution of events has been generated and the results printed in Chronis_Sakos_exp.pdf")

    """ Plotting the poisson distribution """
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)
    n, bins, patches = ax3.hist(data_poisson, 50, (0, 16))
    ax3.set_xlabel("x")
    ax3.set_ylabel("Number of entries")
    fig3.savefig("Chronis_Sakos_poisson.pdf")
    plt.title("Poisson Distribution")
    plt.show()
    plt.close(fig3)    
    print("\tA poisson distribution of events has been generated and result is printed in Chronis_Sakos_poisson.pdf")
