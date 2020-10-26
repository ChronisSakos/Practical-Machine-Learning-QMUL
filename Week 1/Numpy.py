"""
------------------------------------------------------------------------
  author:       Chronis Sakos (c.sakos@stu18.qmul.ac.uk)
  Student ID:   190744335
  Date:         27/07/2019
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

matrix_1 = np.identity(4)
print("Matrix 1 is:\n", matrix_1)

matrix_2 = np.random.random((1,4,4))
print("Matrix 2 is:\n", matrix_2)

sum_0 = matrix_1 + matrix_2                   
""" 
SUM
"""
product = np.matmul(matrix_1,matrix_2)     
"""
PRODUCT
"""
difference = matrix_1 - matrix_2          
""" 
DIFFERENCE
"""

print("The sum of the two matrices is:\n", sum_0)
print("The product of the two matrices is:\n", product)
print("The difference between the matrices is:\n", difference)