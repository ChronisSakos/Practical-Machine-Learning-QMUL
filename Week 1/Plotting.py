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
import matplotlib.pyplot as plt


# this is required to permit multiple copies of the OpenMP runtime to be linked
# to the programme.  Failure to include the following two lines will result in
# an error that Spyder will not report.  On PyCharm the error provided will be
#    OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
#    ...
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



x = []
y = []

for i in range(10):
  x.append(i)
  y.append(i**2) 
  
plt.plot(x, y, 'r--')
plt.ylabel('y[m/s]')
plt.xlabel('x[m]')
plt.title('y=x^2')

plt.show()
