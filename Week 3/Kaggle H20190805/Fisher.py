"""
Fisher.py

Class implementation for a fisher discrimimant.  This class has member functions that
will allow the calculation of Fisher coefficients to separate two classes of event,
denoted by A and B in the implementation.  The Fisher discriminant is given by

   F = alpha^T x + beta

where alpha is a vector of Fisher coefficients (each coeeficient is a scalar), x
is the vector of the features and beta is an arbitrary offset. This class ignores 
beta as it is an arbitrary offset that can be chosen and it contains no information
related to separation between the two classes of event.  Based on the Fisher
discriminant discussed in R. A. Fisher, Ann. Eug., 7, 179188 (1936).

For example, given TF variables of data classA and classB, and a python array of
column labels columns one can compute and print out Fisher coefficients using:

  fish = Fish.Fisher("myfisher")
  fish.CalcCoefficients(classA, classB, columns)
  fish.Print()

Alternatively the data members can be accessed directly, e.g.

  print(fish.alpha)

Having computed fisher coefficients you can use the 
  ProcessData
  CalcVal
functions to process a data set, or a single example respectively in order to compute
the fisher value for that data.

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
import PracticalMachineLearning as PML


class Fisher:
    """
-------------------------------------------------------------------------------------
 Class to facilitate computation of fisher coefficients for an MVA problem.

    alpha = W^-1 (mu[A] - mu[B])

 where alpha is an array of Fisher coefficients, W is the sum of covariance matrices
 and mu[A] and mu[B] are column matricies of hte mean values of data for class A and B.

 This class has the following data members:
   - name      Instance name
   - alpha     Array of Fisher coefficients calculated
   - deltaMu   Array of deltaMu values
   - W         Sum of covariances
   - Winv      Inverse of W
   - Verbose   Printing flag (FALSE by default)
   - sess      TF Session
-------------------------------------------------------------------------------------
    """
    def __init__(self, name="fisher"):
        self.name               = name
        self.alpha              = []
        self.sess               = tf.Session()
        self.deltaMu            = []
        self.W                  = []
        self.Winv               = []
        self.Verbose            = bool(0)

    def CalcCoefficients(self, classA, classB, columns=[]):
        """
---------------------------------------------------------------------------------
 Function to compute the fisher discriminant coefficients given tensors of the
 specified shape corresponding to two classes of event
 [N examples, feature sapace dimension] first reset the arrays to make sure that
 we don't overwrite anything from a previous computation; then proceed with
 computation.

   - classA and classB are tensors of type tf.Variable
   - columns is an array of column names
---------------------------------------------------------------------------------
        """
        del self.alpha[:]
        del self.deltaMu[:]
        del self.W[:]
        del self.Winv[:]

        init = tf.global_variables_initializer()
        self.sess.run(init)
        meanA, varA = tf.nn.moments(classA, axes=[0])
        meanB, varB = tf.nn.moments(classB, axes=[0])

        if(len(columns) == 0):
            # make a dummy set of columns for covariance matrix comp
            row0 = self.sess.run(classA)[0]
            for i in range(len(row0)):
                columns.append(str(i))

        # compute the covariance matrices for the data
        CovA = PML.CalcCovarianceMatrixT(self.sess, classA, columns)
        CovB = PML.CalcCovarianceMatrixT(self.sess, classB, columns)

        Winv = tf.matrix_inverse(tf.add(CovA, CovB))
        DeltaMu = tf.subtract(meanA, meanB)

        # compute coefficients using array form
        self.Winv     = self.sess.run(Winv)
        self.deltaMu  = self.sess.run(DeltaMu)

        for i in range(len(self.Winv)):
            row = self.Winv[i]
            thisAlpha = 0
            for j in range(len(row)):
                thisAlpha += row[j]*self.deltaMu[j]
            self.alpha.append(thisAlpha)

        if(self.Verbose):
            if(len(columns)>0):
                for i in range(len(columns)):
                    print ("\talpha({0}) = {1:.4f}".format(columns[i], self.alpha[i]))
            else:
                for i in range(len(columns)):
                    print ("\talpha({0:d}) = {1:.4f}".format(i, self.alpha[i]))


    def ProcessData(self, data):
      """
---------------------------------------------------------------------------------
Compute the fisher values given the data.  Returns a tensor of rank 1 corresponding
to the Fisher values evaluated for each example in data
---------------------------------------------------------------------------------
      """
      Shape = tf.shape(data)
      Nexample = self.sess.run(Shape)[0]
      Ndim     = self.sess.run(Shape)[1]

# this works but is very slow as it is traditional Python coding that neglects
# the fact that tensor ops can be used more efficiently.
#      FishData = []
#      for i in range(Nexample):
#          example = self.sess.run(data[i, :])
#          val = self.CalcVal(example)
#          FishData.append(val)
#      return FishData

# this approach uses tensor ops and is much faster.  You can test the speed of
# calculations out by commenting this code out and uncommenting the above.
      val = tf.Variable(tf.zeros([Nexample], dtype=tf.float32))
      self.sess.run(val.initializer)
      for j in range(Ndim):
          data_col = data[:, j]
          self.sess.run(val.assign( val + self.alpha[j]*data_col) )

      return val


    def CalcVal(self, dataExample):
        """
---------------------------------------------------------------------------------
 Given the computed coefficients, evaluate the Fisher output for an
 example provided
---------------------------------------------------------------------------------
        """
        thisVal = 0
        if(len(self.alpha) >0):
            for j in range(len(dataExample)):
                thisVal += dataExample[j]*self.alpha[j]
        else:
            print ("""
ERROR - unable to compute Fisher discriminant as coeffients have not yet been computed"

To use the function Fisher::CalcVal, you first have to call CalcCoefficients to evaluate 
the Fisher coefficients.
""")
        return thisVal


    def Print(self):
        """
---------------------------------------------------------------------------------
 Print out the Fisher coefficients, deltaMu and W^-1
---------------------------------------------------------------------------------
        """
        print ("Fisher computation summary:")
        if(len(self.alpha) == 0):
            print ("""
ERROR - unable to display Fisher discriminant as coeffients have not yet been computed"

To use the function Fisher::Print, you first have to call CalcCoefficients to evaluate 
the Fisher coefficients.
""")
            return

        print ("Delta Mu column tensor:")
        for i in range(len(self.deltaMu)):
            print ("\tDeltaMu({0:d}) = {1:.4f}".format(i, self.deltaMu[i]))

        print ("W^-1 tensor:")
        PML.PrintMatrix(self.Winv)

        print ("\nalpha coefficients:")
        for i in range(len(self.alpha)):
            print ("\talpha({0:d}) = {1:.4f}".format(i, self.alpha[i]))

