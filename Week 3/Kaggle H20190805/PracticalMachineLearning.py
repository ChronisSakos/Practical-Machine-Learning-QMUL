"""
PracticalMachineLearning.py

This python script contians a set of helper functions for use with the PracticalMachineLearning
course used for the Queen Mary University of London Summer School.

Please see 
   https://qmplus.qmul.ac.uk/course/view.php?id=10006

for more details of the course.

  ----------------------------------------------------------------------
  author:       Adrian Bevan (a.j.bevan@qmul.ac.uk)
  Copyright (C) QMUL 2018
  ----------------------------------------------------------------------
"""

import math
import tensorflow as tf

def PrintMatrix(matrix, format="normal"):
  """
  ----------------------------------------------------------------------
   Function to print out a covariance/correlation matrix for a 2D array
   
   matrix          - a python array of the type produced by CalcCovarianceMatrix 
                     of CalcCorrelationMatrix
   format          - A string specifying output format:
                         normal  : standard text printout to stdio
                         tex     : LaTeX formatted table printout to stdio

  ----------------------------------------------------------------------
  """
  if(format == "normal"):
    for i in range(len(matrix)):
      string = "\t"
      for j in range(len(matrix)):
        string += "{:.2f}".format(matrix[i][j])
        string += " "
      print (string)
  elif((format == "tex") or (format == "latex")):
    print ("\\begin{table}[!ht]")
    print ("\\caption{Matrix caption to complete}\n\\label{tbl:dummy}")
    printstring = "\\begin{tabular}{"
    for i in range(len(matrix)):
      printstring += "l"
    printstring += "}\n\\hline\\hline"
    print (printstring)
    for i in range(len(matrix)):
      string = "\t"
      for j in range(len(matrix)):
        string += "{:.2f}".format(matrix[i][j])
        string += " & "
      string += "\\\\"
      print (string)
    print ("\\hline\\hline\n\\end{tabular}\n\\end{table}")
  return



def CalcCovariance(Data, col1, col2):
  """
  ----------------------------------------------------------------------
   Compute the covariance (and correlation) between tensor columns col1 and col2
  
   Data       - this is a python array
   col1       - the first index (this is an integer)
   col2       - the second index (this is an integer)

  The covariance is given by 
           cov(col1, col2) = sum_{data} (col1 - col1mean)(col2-col2mean)
                                        --------------------------------
                                                     N-1
  Also computes the Pearson correlation given by
           corr(col1, col2) = cov(col1, col2) / sigma(col1)sigma(col2)

  ----------------------------------------------------------------------
  """
  # need to compute the mean
  MeanCol1 = 0
  MeanCol2 = 0
  for i in range(len(Data)):
    ThisExample = Data[i]
    MeanCol1 += ThisExample[col1]
    MeanCol2 += ThisExample[col2]

  MeanCol1 = MeanCol1/len(Data)
  MeanCol2 = MeanCol2/len(Data)

  # now compute the sum of the residuals (squared)
  sResCol1 = 0
  sResCol2 = 0
  sResProd = 0
  for i in range(len(Data)):
    ThisExample = Data[i]
    ResCol1 = (ThisExample[col1] - MeanCol1)
    ResCol2 = (ThisExample[col2] - MeanCol2)

    sResCol1 += ResCol1*ResCol1
    sResCol2 += ResCol2*ResCol2
    sResProd += ResCol1*ResCol2

  # take the residual sums (squared) and convert those to variances
  sResCol1 = sResCol1/(len(Data)-1)
  sResCol2 = sResCol2/(len(Data)-1)
  sResProd = sResProd/(len(Data)-1)

  covar = sResProd
  corr  = sResProd/math.sqrt(sResCol1)/math.sqrt(sResCol2)

  return covar, corr



def CalcCovarianceT(sess, DataTensor, col1, col2):
  """
  ----------------------------------------------------------------------
   Compute the covariance (and correlation) between tensor columns col1 and col2
  
   DataTensor - this is a tensor
   col1       - the first index (this is an integer)
   col2       - the second index (this is an integer)
  ----------------------------------------------------------------------
  """
  Data = sess.run(DataTensor)
  covariance = CalcCovariance(Data, col1, col2)

  return covariance


def CalcCovarianceMatrix(Data, columns):
  """
  ----------------------------------------------------------------------
   Compute the covariance matrix for all combinations of pairs of variables
  
   Data - this is a python array
   this is a python list of column headings
  ----------------------------------------------------------------------
  """
  covMatrix=[]#create an empty list first
  for i in range(len(columns)):
    covMatrix.append([0]*len(columns))
    for j in range(len(columns)):
      covMatrix[i][j] = 0.0

  for i in range(len(columns)):
    for j in range(len(columns)):
      if(i>=j):
        covMatrix[i][j], dummy = CalcCovariance(Data, i, j)

  for i in range(len(columns)):
    for j in range(len(columns)):
      if(i>j):
        covMatrix[j][i] = covMatrix[i][j]

  return covMatrix

def CalcCovarianceMatrixT(sess, DataTensor, columns):
  """
  ----------------------------------------------------------------------
   Compute the covariance matrix for all combinations of pairs of variables
  
   DataTensor - this is a tensor
   this is a python list of column headings
  ----------------------------------------------------------------------
  """
  Data = sess.run(DataTensor)
  covarianceMatrix = CalcCovarianceMatrix(Data, columns)

  return covarianceMatrix



def CutOutNegativeData(sess, TFVdata, columns, columnToCut):
    """
    ----------------------------------------------------------------------
     Apply a cut on the data for the specified column, retain values that
     are >= 0, and reject all negative values.  For example this helper
     function can be used to remove examples with negative weights prior
     to plotting/training etc.
    ----------------------------------------------------------------------
    """
    # get the column index
    thisColIndex = -999
    for i in range(len(columns)):
        if columnToCut == columns[i]:
            thisColIndex = i
            break

    if thisColIndex == -999:
        print ("ERROR - Unable to locate the column ", columnToCut, " in order to remove examples with negative values")
        return TFVdata

    init = tf.global_variables_initializer()
    sess.run(init)
    thisData = sess.run(TFVdata)
    newData = []

    for i in range(len(thisData)):
        thisExample = thisData[i]
        if (thisExample[thisColIndex] >= 0):
            thisRow = []
            for j in range(len(thisExample)):
                thisRow.append( thisExample[j] )
            newData.append(thisRow)

    newDataTensor = tf.Variable(newData, name = "reduced_data")

    return newDataTensor




def ExtractColumnTensor(sess, TFVdata, columns, column):
    """
    ----------------------------------------------------------------------
     create a tensor corresponding to a single column in the data. The inputs
     are the session (sess), a data tensor (TFVdata), a list of columns of
     interest (columns) and a string of the column to extract (column).

     This function creates a new tensor, and the tensor needs to be initialised
     before it can be used.
    ----------------------------------------------------------------------
    """
    newdata = ExtractColumn(sess, TFVdata, columns, column)
    thisname = "column_data_"
    thisname += str(column)
    data = tf.Variable( newdata, name=thisname )

    return data


def ExtractColumnsTensor(sess, TFVdata, columns, subcolumns):
    """
    ----------------------------------------------------------------------
     create a tensor corresponding to specified columns of the data. The inputs
     are the session (sess), a data tensor (TFVdata), a list of columns of
     interest (columns) and a list of the columns to extract (subcolumns).

     This function creates a new tensor, and the tensor needs to be initialised
     before it can be used.
    ----------------------------------------------------------------------
    """
    newdata = ExtractColumns(sess, TFVdata, columns, subcolumns)
    thisname = "column_data_"
    for col in subcolumns:
        thisname += str(col)
        thisname += "_"

    data = tf.Variable( newdata, name=thisname )

    return data

def ExtractColumns(sess, TFVdata, columns, subcolumns):
    """
    ----------------------------------------------------------------------
     extract a subspace of the input feature space from the data
     the index of the tensor data to extract.

        sess        - the session
        TFVdata     - A TensorFlow tensor of data with dimensions [N, dim(x)]
        columns     - the column names
        subcolumns  - the name of the columns to extract from the data

    ----------------------------------------------------------------------
    """
    # loop over the subcolumns to find the correspinding indices that match the columns.
    colNumbers = []
    for i in range(len(subcolumns)):
        counter = 0
        for col in columns:
            if(subcolumns[i] == col):
                colNumbers.append(counter)
                break
            counter += 1

    # now extract the data in the appropriate ordering
    dataToProcess = sess.run(TFVdata)
    N = len(dataToProcess)
    print ("Extracting column ", colNumbers, " from the data. There are N = ", N+1, " examples in this data set")
    print ("The extracted columns correspond to the feature labels ", subcolumns)

# Convert the tensor back to an array so only run the session once; otherwise the
# extraction of columns is inefficient

    thisDataSubset = []
    for i in range( N ):
        # create a copy of the example data that will only contain the examples in the
        # order presented by the subcolumns variable.
        thisExample = []
        for j in range( len(subcolumns) ):
            thisExample.append( dataToProcess[i][ colNumbers[j] ] )

        thisDataSubset.append( thisExample )
        if not ((i+1) % 50000):
            print ("\tProcessed ", i+1, " examples")

    return thisDataSubset




def ExtractColumn(sess, TFVdata, columns, column):
    """
    ----------------------------------------------------------------------
     extract a column of data from the tensor; using the column name to identify
     the index of the tensor data to extract.

        sess        - the session
        TFVdata     - A TensorFlow tensor of data with dimensions [N, dim(x)]
        columns     - the column names
        column      - the name of the column to extract from the data

    ----------------------------------------------------------------------
    """
    columnstring = ""
    counter = 0
    colNumber = -1
    thisColumn = []
    for col in columns:
        if(column == col):
            colNumber = counter
            break
        counter += 1

    if colNumber < 0:
        print ("ERROR - unable to extract column ", column, " from the data sample; returning a null list")
        return thisColumn

    thisData = sess.run(TFVdata)
    N = len(thisData)
    print ("Extracting column ", colNumber, " (", column, ") from the data. There are N = ", N+1, " examples in this data set")

    for i in range( N ):
        thisColumn.append( thisData[i][colNumber]  )

    return thisColumn




def ReadCSVasTFV(filename, option=""):
    """
    ----------------------------------------------------------------------
     read the CSV file as a tensor flow variable, use ReadCSV to import
     the data as an array, and then convert to a TF variable. Requires a
     session to be running.

     To use this function you first call the function, and then need to
     start the session and initilise the graph using, for example:

         data, columns, stdata, stcolumns = hipster.ReadCSVasTFV("testdata.csv")

         sess = tf.Session()
         init = tf.global_variables_initializer()
         sess.run(init)

         # no the tf variable can be used, this will have shape [Ndata, dim(x)]
         print sess.run(data3tf)

     This function returns a TensorFlow Variable for the data set and a
     python list for the column names.
    ----------------------------------------------------------------------
    """
    data, columns, stringdata, stringcolumns = ReadCSV(filename, option)
    dataTensor = tf.Variable(data, name = "data_from_csv_file")
    stringDataTensor = tf.Variable(data, name = "string_data_from_csv_file")

    return dataTensor, columns, stringDataTensor, stringcolumns




def parseOptionsIO(option, VERBOSE, QUIET):
    """
    ----------------------------------------------------------------------
     this is a helper function for I/O options
    ----------------------------------------------------------------------
    """
    if(option == "v" or option == "verbose"):
        VERBOSE = 1
    if(option == "q" or option == "quiet"):
        QUIET = 1

    return VERBOSE, QUIET



def ReadCSV(filename, option=""):
    """
    ----------------------------------------------------------------------
     Read in the specified CSV file; the first line is a comma separated list of
     columns to be read into the CSV file. The dimensionality of that sets the
     dimensionality of the data.  If events do not match the column heading
     dimensionality then an error is printed out and the event is skipped.

     This function returns:
       theData       : a list of lists / the data (floating point representation)
       theDataColumns: a list of column names
       theStringData : a list of lists of data for string based information
                       (e.g. string labels)
       theStringDataColumns: a list of column names
    ----------------------------------------------------------------------
    """
    print ("Will read the file ", filename)

    # strip the newline character from the end of the line, and then split the data to create columns
    data    = open(filename).readlines()
    data[0] = data[0].strip()
    columns = data[0].split(',')

    VERBOSE = bool(0)
    QUIET   = bool(0)
    VERBOSE, QUIET = parseOptionsIO(option, VERBOSE, QUIET)

    if VERBOSE:
        print ("  This file contains the following columns")
        for column in columns:
            print ("    ", column)
        print ("  and contains ", len(data), " entries")
    elif not QUIET:
        print ("  This file contains ", len(columns), " columns and ", len(data), " entries")

    # loop over the data, for each line split the data on commas,
    # suppress white space characters and convert to numbers
    # to store in an array for.  The variable theData is the
    # numerical data of interest, and will be returned to the
    # calling function.
    theData = []
    theStringData = []
    skippedEvents = 0
    theDataColumns = []
    theStringDataColumns = []
    for i in range(len(data)-1):
        thisRow = []
        thisStringRow = []
        thisLine = data[i+1]
        thisLine = thisLine.replace(" ", "")
        csvLineData = thisLine.split(',')
        if len(csvLineData) != len(columns):
            print ("\tERROR READING LINE ", i, " (wrong number of columns for this row, found ", len(csvLineData), " expected ", len(columns), " )")
            skippedEvents+=1
        else:
            for j in range(len(csvLineData)):
                try:
                    float(csvLineData[j])
                    thisRow.append( float(csvLineData[j]) )
                    if i == 0:
                        theDataColumns.append(columns[j])
                except ValueError:
                    thisStringRow.append( csvLineData[j] )
                    if i == 0:
                        theStringDataColumns.append(columns[j])
            if VERBOSE :
                print (thisRow)

            # append the data samples with lvalue and string features
            if(len(thisRow)):
                theData.append(thisRow)
            if(len(thisStringRow)):
                theStringData.append(thisStringRow)

    if(skippedEvents):
        print ("\tWARNING: a total of ", skippedEvents, " have been skipped when reading this data in - data file is not properly formatted")

    if not QUIET:
        print ("\tThe file has been read")

    return theData, theDataColumns, theStringData, theStringDataColumns



def MergeFeatureSets(sFeatureSpace, bFeatureSpace, sLabels, bLabels, sWeights, bWeights, NtoMerge=-999, NormalisationType="[0,1]"):
    """
    ----------------------------------------------------------------------
     Given input data sets corresponding to signal and background examples,
     construct an ensemble sample that interleaves the signal and background
     training examples; ensure that labels and weights are adequately
     tracked into the new sample.

      sFeatureSpace  - signal feature space
      bFeatureSpace  - background feature space
      sLabels        - signal labels
      bLabels        - background labels
      sWeights       - signal weights
      bWeights       - background weights
      NtoMerge       - the number of events to merge; if -999 then use the largest possible numbers
                       given the sample sizes, where equal example numbers are retained.
      NormalisationType
                     - default [0,1]: map features onto this range
                     - none: do not map features
                     - [-1,1]: map features onto this range

     This function will return three variables; the data, weights and
     labels (all prepared as tensors)
    ----------------------------------------------------------------------
    """
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    sDat = sess.run(sFeatureSpace)
    bDat = sess.run(bFeatureSpace)
    sL   = sess.run(sLabels)
    bL   = sess.run(bLabels)
    sW   = sess.run(sWeights)
    bW   = sess.run(bWeights)

    Nsig = len(sDat)
    Nbg  = len(bDat)
    N = Nsig
    if(Nsig > Nbg):
        N = Nbg
    if (NtoMerge >0) and (NtoMerge < 2.0*N):
        N = int(NtoMerge/2.0)

    print ("MergeFeatureSets: Will construct a merged sample using ", N, " examples")
    print ("                  from the signal and background data sets.")
    # use the smallest number of examples as the number to use from either sample.
    newData    = []
    newLabels  = []
    newWeights = []
    for i in range(N):
        # do element-wise copy of the data for tensor construction
        sigRow = []
        bgRow = []
        for val in sDat[i]:
            sigRow.append(val)
        for val in bDat[i]:
            bgRow.append(val)

        newData.append( sigRow )
        newData.append( bgRow )

        newLabels.append( [1] )
        newLabels.append( [0] )

        newWeights.append( sW[i] )
        newWeights.append( bW[i] )

    print ("Created a merged data sample with ", len(newData), " examples from signal and background")
    newDataTensor    = tf.Variable(newData,    name = "merged_data")
    newLabelsTensor  = tf.Variable(newLabels,  name = "merged_labels")
    newWeightsTensor = tf.Variable(newWeights, name = "merged_weights")

    # check to see if we need to transform the data or not.  By deafault the feature
    # space is normalised onto the coordinate range [0,1] for all variables.
    # The following mapping options are:
    #    [0,1]            <---- default behaviour
    #    [-1,1]
    #    none
    init = tf.global_variables_initializer()
    sess.run(init)
    newData = sess.run(newDataTensor)
    if(NormalisationType == "[0,1]"):
        thisData = NormaliseData(newData)
        newDataTensor = tf.Variable(thisData, "normalised_merged_data")
        print ("Merged data sample has been created - feature space is mapped into the range [0, 1]")
    elif (NormalisationType == "[-1,1]"):
        thisData = NormaliseDataNeg1To1(newData)
        newDataTensor = tf.Variable(thisData, "normalised_merged_data")
        print ("Merged data sample has been created - feature space is mapped into the range [-1, 1]")
    elif (NormalisationType == "none"):
        print ("Merged data sample has been created - feature space is not mapped")
    init = tf.global_variables_initializer()
    sess.run(init)

    print ("Merged data tensor shape is:")
    print (newDataTensor)

    return newDataTensor, newLabelsTensor, newWeightsTensor



def NormaliseData(data, MinRange=[], MaxRange=[]):
    """
    ----------------------------------------------------------------------
     Apply the transformation: x' = ax + b to each column of data.
     The column ranges are mapped to [0, 1] unless specified otherwise.
    ----------------------------------------------------------------------
    """
    # determine the number of features in the data
    N = len(data[0])

    a = []
    b = []

    # compute the normalisation ranges for all columns of data
    for i in range(N):
        thisA, thisB = GetFeatureRange(data, i)
        a.append(thisA)
        b.append(thisB)

    # ax+b maps the data onto [0, 1], if we want to map onto [newMin, newMax]
    # therefore we have thisRange*(ax+b) + newMin as the function to map
    # x = [min, max] to x'' = [newMin, newMax].  For now just map to [0, 1]
    # for all data.
    newData = []
    if(len(MinRange) == 0) and (len(MaxRange)==0):
        # scale normalisation to [0, 1] for all columns
        for i in range(len(data)):
            thisRow = data[i]
            theNewRow = []
            for j in range(len(thisRow)):
                theNewRow.append( thisRow[j] * a[j] + b[j] )
            newData.append(theNewRow)
    else:
        # scale normalisation according to ranges supplied by the calling function
        for i in range(len(data)):
            thisRow = data[i]
            theNewRow = []
            for j in range(len(thisRow)):
                thisMin = MinRange[j]
                thisMax = MaxRange[j]
                thisRange = thisMax-thisMin
                theNewRow.append( thisRange * (thisRow[j] * a[j] + b[j] ) + thisMin )
            newData.append(theNewRow)

    return newData

def NormaliseDataNeg1To1(data):
    """
    ----------------------------------------------------------------------
     Apply the transformation: x' = ax + b to each column of data to map
     column ranges to [-1.0, 1.0] for test/training/evaluation purposes
    ----------------------------------------------------------------------
    """
    FirstExample = data[0]
    MinRange = []
    MaxRange = []
    for i in range(len(FirstExample)):
        MinRange.append( -1.0 )
        MaxRange.append( 1.0 )

    theNewData = NormaliseData(data, MinRange, MaxRange)

    return theNewData


def NormaliseColumn(data, newMin=0, newMax=1):
    """
    ----------------------------------------------------------------------
     Apply the transformation: x' = ax + b to a column of data.  If newMin
     and newMax are supplied then the mapping is to the following value
      x'' = (ax + b)*(newMax-newMin) + newMin; where
         x has the range [min, max] given by data
         x' has the range [0, 1]
         x'' has the range [newMin, newMax]
    ----------------------------------------------------------------------
    """
    if(newMax == newMin):
        print ("unable to normalise data to a distribution with zero range")
        return data

    # compute the normalisation range given this is a column, the feature
    # index is 0.
    a, b = GetFeatureRange(data, 0)

    # ax+b maps the data onto [0, 1], we want to map onto [newMin, newMax]
    # therefore we have thisRange*(ax+b) + newMin as the function to map
    # x = [min, max] to x'' = [newMin, newMax]
    thisRange  = newMax-newMin
    thisOffset = newMin

    newData = []
    for i in range(len(data)):
        thisRow = data[i]
        theNewRow = []
        theNewRow.append( thisRange*(thisRow[0] * a + b) + newMin )
        newData.append(theNewRow)

    return newData


def GetFeatureMaxMin(data, columnIndex):
    """
    ----------------------------------------------------------------------
     Given a feature in some data find the maximum and minimum values of that
     feature.
    ---------------------------------------------------------------------
    """
    MinRange = float(1e10)
    MaxRange = float(-1e10)

    for i in range(len(data)):
        thisRow = data[i]
        if(thisRow[columnIndex] < MinRange):
            MinRange = thisRow[columnIndex]
        if(thisRow[columnIndex] > MaxRange):
            MaxRange = thisRow[columnIndex]

    if (MaxRange == MinRange):
        print ("Maximum and minmum range indices match: [", MinRange, ", ", MinRange, "]")
        return 1.0, 0.0

    return MaxRange, MinRange



def GetFeatureRange(data, columnIndex):
    """
    ----------------------------------------------------------------------
     Given a feature in some data find the maximum and minimum values of that
     feature.  The linear transformation a*x+b can be determined from the range
     as x / (max-min) - min / (max-min).  This function computes the scale and
     offset required to map some feature space dimension [min, max] onto [0, 1]
     by determining the parameters a and b. These are returned to the calling
     function.

     If the range, max-min, is found to be zero then a=1.0 and b = 0 is
     returned.

     The transformation from [min, max] onto [0, 1] is achieved by
     computing:
                             x' = ax + b
    ----------------------------------------------------------------------
    """
    MaxRange, MinRange = GetFeatureMaxMin(data, columnIndex)
    Normalisation      = 1.0 / (MaxRange - MinRange)
    Offset             = -1.0 * MinRange * Normalisation

    return Normalisation, Offset
