"""
------------------------------------------------------------------------
  author:       Chronis Sakos (c.sakos@stu18.qmul.ac.uk)
  Student ID:   190744335
  Date:         24/07/2019
  ----------------------------------------------------------------------
"""

import math
import numpy as np
import tensorflow as tf
import matplotlib as plt

# this is required to permit multiple copies of the OpenMP runtime to be linked
# to the programme.  Failure to include the following two lines will result in
# an error that Spyder will not report.  On PyCharm the error provided will be
#    OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
#    ...
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
2 scalars
"""
a        = tf.constant(3.0, dtype=tf.float32, name="a")
b        = tf.constant(4.0, name="b") 

total    = a + b
difference = a - b
product  = a * b
ratio = a/b

"""
2 vectors
"""
c        = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=tf.float32)  
d        = tf.constant([1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1, 10.0], dtype=tf.float32)

total1 = tf.add(c,d)
difference1 = tf.subtract(c,d)
product1 = tf.multiply(c,d)
ratio1 = tf.div(c,d)

""" 
2 matrices
"""
e        = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32) 
f        = tf.constant([[2, 4, 6], [8, 10, 1], [3, 5, 7]], dtype=tf.float32)  

total2 = tf.add(e,f)
difference2 = tf.subtract(e,f)
product2 = tf.multiply(e,f)
ratio2 = tf.div(e,f)


print("Graph structure (names, shapes, types)")

print("a           = ", a)
print("b           = ", b)
print("Total       = ", total)
print("Product     = ", product)
print("Difference  = ", difference)
print("Ratio       = ", ratio) 
 
print("Total_vect       = ", total1)
print("Product_vect     = ", product1)
print("Difference_vect  = ", difference1)
print("Ratio_vect       = ", ratio1)

print("Total_mat        = ", total2)
print("Product_mat      = ", product2)
print("Difference_mat   = ", difference2)
print("Ratio_mat        = ", ratio2)

sess = tf.Session()

print("\nGraph calculations (evaluation)")
print("a                    = ", sess.run(a))
print("b                    = ", sess.run(b))
print("Total                = ", sess.run(total))
print("Product              = ", sess.run(product))
print("Difference           = ", sess.run(difference))
print("Ratio                = ", sess.run(ratio))

print("Total_vect           = ", sess.run(total1))
print("Product_vect         = ", sess.run(product1))
print("Difference_vect      = ", sess.run(difference1))
print("Ratio_vect           = ", sess.run(ratio1))

print("Total_mat            = ", sess.run(total2))
print("Product_mat          = ", sess.run(product2))
print("Difference_mat       = ", sess.run(difference2))
print("Ratio_mat            = ", sess.run(ratio2))


initial_value = tf.zeros([10,3], tf.int32, name="initial_value")
tensorVar = tf.Variable(initial_value, name="tensorVar")
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print("Variable tensor with shape [10,3] initialized to zeros\n", sess.run(tensorVar))

for i in range(10):
    a = tf.assign(tensorVar[i,0],i)
    b = tf.assign(tensorVar[i,1],i**2)
    c = tf.assign(tensorVar[i,2],i**3)
    sess.run(a)
    sess.run(b)
    sess.run(c)
    
print("Updated Variable tensor\n", sess.run(tensorVar))
