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


class MyComplexNumber:
    """
    This is a complex number representation in python as an example of how to write a class.
    """
    a = 0.0
    b = 0.0
   
    def __init__(self, thisa=0.0, thisb=0.0):
        self.a = thisa
        self.b = thisb
        
    def update_a(self):
        print("Give the value of a:")
        newa = float(input())        
        """
        input value for a
        """
        self.a=newa
        
    
    def update_b(self):
        print("Give the value of b:")
        newb = float(input())  
        """
        input value for b
        """
        self.b=newb
        
        
    def new_r_and_theta(self):
            print("Give the value of r:")
            r = float(input())         
            """input for r and theta"""
            print("Give the value of theta (rad):")
            th = float(input())
            self.a = r*math.cos(th)     
            """
            updating a and b relating to the values of r and theta
            """
            self.b = r*math.sin(th)
            
  
    def magnitude(self):
        """
        r = |z| = sqrt(a^2 + b^2) = sqrt(zz*)
        """
        return math.sqrt(self.a**2 + self.b**2)

    def real(self):
        """
        z = a + i b, where a is the real part of the complex number and b is the imaginary part
        """
        return self.a

    def imaginary(self):
        """
        z = a + i b, where a is the real part of the complex number and b is the imaginary part
        """
        return self.b

    
    def phase(self):
        """
        z = a + i b, where a is the real part of the complex number and b is the imaginary part.
        The phase is theta for the representation z = r e^{i theta}
        """
        if self.a == 0:
            if self.b > 0:
                return math.pi/2
            else:
                return -math.pi/2           
        return math.atan(self.b/self.a)
        
       
z = MyComplexNumber()


x = None
while x is None:
    while x != 1 or x != 0:
        input_value = input("if you want to give a and b as input type 1 else if you want to give r an theta type 0\n")
        try:
            x = int(input_value)
            """
            we ask the user to choose whether to give input for a and b OR r and theta
            """
            
            if x==1:        
                """
                input for a and b
                """
                z.update_a()
                z.update_b()
                print("\tResults:")
                print("|z|    = ", z.magnitude())
                print("Re(z)  = ", z.real())
                print("Im(z)  = ", z.imaginary())
                print("arg(z) = ", z.phase())
                break
                    
            elif x==0:      
                """
                input for r and theta
                """
                z.new_r_and_theta()
                print("|z|    = ", z.magnitude())
                print("Re(z)  = ", z.real())
                print("Im(z)  = ", z.imaginary())
                print("arg(z) = ", z.phase())
                break
            else:
                print("invalid input, try again")   
                """ 
                wrong input
                """
        except ValueError:
            print("{input} is not a number, please enter 1 or 0".format(input=input_value))
     

                    
    

 

  

