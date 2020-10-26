#!/usr/bin/python
#
# Test the TensorFlow api using the ATLAS Higgs challenge data with a deep
# network.  This script will read in training data and prepare a merged
# sample of signal and background (alternating between the two). A DNN
# will be trained using the ADAM optimiser and the softmax actition function.
#
######################################################################
# TensorFlow test example prepared by Adrian Bevan (a.j.bevan@qmul)
# for use with the material found at:
#
# https://pprc.qmul.ac.uk/~bevan/statistics/TensorFlow.html
#
# July 2018
######################################################################
import tensorflow as tf
import matplotlib.pyplot as plt
import PracticalMachineLearning as PML
import numpy as np

# global network configuration
learning_rate      = 0.001
training_epochs    = 2000
n_output           = 1
NTrainingData      = 5000
DropOutKeepProb    = 0.9

# Deep network architecture parameters go here.  We start with a simple
# 1 hidden layer neural network so that this can be developed into a 
# more complex model
n_hidden_1 = 256      # 1st layer num features

def RunAnalysis():
  #----------------------------------------------------------------------
  # Main entry point to the network model
  #----------------------------------------------------------------------

    print ("""
--------------------------------------------------------------------------------
Kaggle Analysis                                                                V1.0

  This script will construct a NN to solve a 2 output target classification 
  problem based on input data from the H->tau+tau- Kaggle data set.  The original
  skeleton has a single hidden layer with 256 nodes and this will be extended
  to more complex modules by students working on the problem.

--------------------------------------------------------------------------------
""")
# extract signal and background features - the feature set is built in a consistent
# way for each sample; removing the negative weight events, and selecting columns
# of interest for the analysis.  The signal and background feature spaces
# are the same; the examples, labels and weights are drawn from the data after
# the cut on weights.
#
# - The feature space for s(ignal) and b(ackground) is a tensor
# - Features for s and b are lists
# - Labels for s and b are tensors
# - Weights for s and b are tensors
    print ("  Extracting signal data ........................")
    sFeatureSpace, sFeatures, sLabels, sWeights = BuildFeatureSpace("train_sml_sig.csv")
    print ("  Extracting background data ....................")
    bFeatureSpace, bFeatures, bLabels, bWeights = BuildFeatureSpace("train_sml_bg.csv")

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    print("""

Data sets have been extracted.  The following features have been obtained as a subsample of the
input data:

""")
    for feature in sFeatures:
        print(feature)


    print ("\n\n Will create merged data sample; where we interleave signal and background training examples\n\n")
#===================================================================================================================
# by default when the features are merged the feature space will be mapped onto the domain [-1, 1] in each dimension.
# This rescaling is done in order to ensure that the L values for features are the same order of magnitude. Otherwise
# the training output can be affected by large differences in scales (e.g. TeV vs GeV vs MeV vs a jet flavour tag).
#
# The labels are assigned as follows: signal = +1, background = -1
#===================================================================================================================
    FeatureSpace, Labels, Weights = PML.MergeFeatureSets(sFeatureSpace, bFeatureSpace, sLabels, bLabels, sWeights, bWeights, NTrainingData, "[-1,1]")

    # ensure the new tf variables are initialised
    sess.run(FeatureSpace.initializer) 
    sess.run(Labels.initializer) 
    sess.run(Weights.initializer)

    print ("""

Set up and train the model using the merged feature space containing signal and background examples

""")

#===================================================================================================================
# setup the model, recall that the architecture in terms of number of outputs and hidden layer nodes
# are defined at the start of this script.
#===================================================================================================================
    n_input = len(sFeatures)
    x         = tf.placeholder(tf.float32, [None, n_input], name="x")    # data placeholder
    y         = tf.placeholder(tf.float32, [None, n_output], name="y")   # label placeholder
    keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

    print("Creating a hidden layer with ", n_hidden_1, " nodes")
    w_layer_1    = tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="WeightsForLayer1")
    bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]), name="BiasForLayer1")
    layer_1      = tf.nn.relu(tf.add(tf.matmul(x,w_layer_1),bias_layer_1))
    dlayer_1 = tf.nn.dropout(layer_1, keep_prob)

    # Similarly we now construct the output of the network, where the output layer
    # combines the information down into a space of evidences for the possible
    # classes in the problem (n_outout=1 for this regression problem).
    print("Creating the output layer ", n_output, " output values")
    output       = tf.Variable(tf.random_normal([n_hidden_1, n_output]), name="OutputNodeWeights")
    bias_output  = tf.Variable(tf.random_normal([n_output]), name = "OutputNodeBias")
    output_layer = tf.nn.sigmoid(tf.matmul(dlayer_1, output) + bias_output)

    # optimise with l2 loss function; this is the cost to compare models with
    print("Using the L2 loss function implemented in tf.nn")
    cost = tf.nn.l2_loss(output_layer - y)
    #cost = tf.nn.l2_loss(output_layer - y) + 0.01*tf.nn.l2_loss(output) + 0.01*tf.nn.l2_loss(w_layer_1)

    print("Using the Adam optimiser to train the network")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    probabilities = output_layer
    traindata     = sess.run(FeatureSpace)
    target_value  = sess.run(Labels)

    print (FeatureSpace)
    print (Labels)

    # initiaialize all model variables - do this the lazy way by running the global initializer just before
    # we start training the model.
    sess.run( tf.global_variables_initializer() )

    training_epoch     = []
    training_cost      = []
    training_accuracy  = []

    # alow for computation of accuracy of the model for model comparison.  The prediction of the 
    # model depends on the label range.  This is 1 for signal and 0 for background, so anything above 
    # 0.5 is signal and so on.
    predictions = tf.cast(output_layer> 0.5, tf.float32)
    correct_prediction = tf.equal( y, predictions )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#===================================================================================================================
# train the model
#===================================================================================================================
    for epoch in range(training_epochs):
        the_cost = 0.

        if not epoch % 50:
            print("Training epoch number ", epoch)
        sess.run(optimizer, feed_dict={x: traindata, y: target_value, keep_prob: DropOutKeepProb})
        the_cost = sess.run(cost, feed_dict={x: traindata, y: target_value, keep_prob: 1.0})

        # log evolution of training as a function of epoch
        training_epoch.append(epoch)
        training_cost.append(the_cost)
        acc =  sess.run(accuracy, feed_dict={x: traindata, y: target_value, keep_prob: 1.0}) 
        training_accuracy.append(acc)

    # get model predicitons and separate these into signal and background outputs. These
    # shapes should be plotted for comparison
    sigPred = []
    bgPred = []
    preddata = sess.run(output_layer, feed_dict={x: traindata, keep_prob: 1.0})
    print("Separating predictions into signal and background")
    for i in range(NTrainingData):
        thisdata = preddata[i][0]
        if target_value[i] == 0:
            bgPred.append( thisdata )
        else:
            sigPred.append( thisdata )

    # plot the output predictions for this model
    fig = plt.figure()
    bins = np.linspace(0, 1, 100)
    plt.hist(sigPred, bins, alpha=0.5)
    #plt.hist(bgPred,  bins, alpha=0.5)
    plt.xlabel('MLP Output Prediction-SIGNAL')
    plt.ylabel('Number of examples')
    plt.show()
    fig.savefig("Example_KaggleHiggs.pdf")

    # plot the output predictions for this model
    fig = plt.figure()
    bins = np.linspace(0, 1, 100)
    #plt.hist(sigPred, bins, alpha=0.5)
    plt.hist(bgPred,  bins, alpha=0.5)
    plt.xlabel('MLP Output Prediction-BACKGROUND')
    plt.ylabel('Number of examples')
    plt.show()
    fig.savefig("Example_KaggleHiggs.pdf")

    # now plot the cost and accuracy for this MLP
    plt.plot(training_epoch, training_cost)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()
    plt.clf()
    plt.plot(training_epoch, training_accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy on training set")
    plt.show()

    print ("""
--------------------------------------------------------------------------------

DNN Model Training has finished.

--------------------------------------------------------------------------------
""")
    print("accuracy = {}".format(training_accuracy[-1]))
    print("[for {} training epochs]".format(training_epoch[-1]+1))

    return




def BuildFeatureSpace(filename, weightName="Weight", labelName="Label"):
#----------------------------------------------------------------------
#
# Read in the data and prepare the feature space for analysis
#
# Extract signal and background features - the feature set is built in a consistent
# way for each sample; removing the negative weight events, and selecting columns
# of interest for the analysis.  The signal and background feature spaces
# are the same; the examples, labels and weights are drawn from the data after
# the cut on weights.
#
#----------------------------------------------------------------------
    KaggleData, columns, StringData, stringColumns = PML.ReadCSVasTFV(filename)

    # remove events with negative weights before building the feature space
    # of interest for the analysis
    sess = tf.Session()
    ReducedKaggleData = PML.CutOutNegativeData(sess, KaggleData, columns, 'Weight')
    ReducedKaggleData = KaggleData
    init = tf.global_variables_initializer()
    sess.run(init)

    print ('\n\nHave read ', len(sess.run(KaggleData))+1, ' examples from the data set ', filename)
    print ('Reduced data set (after removing negative weights) has ', len(sess.run(ReducedKaggleData))+1, ' examples\n\n')
    print ('The following columns (features) have been extracted from the data file.  A subset will be used for analysis')
    print (columns)

    # specify the Kaggle list of variables to use
    Features = []

##=======================================
## !!! NEVER USE EventId AS A FEATURE !!!
##    Features.append('EventId')
##=======================================
    Features.append('DER_mass_MMC')
    Features.append('DER_mass_transverse_met_lep')
    Features.append('DER_mass_vis')
#    Features.append('DER_pt_h')
#    Features.append('DER_deltaeta_jet_jet')
#    Features.append('DER_mass_jet_jet')
#    Features.append('DER_prodeta_jet_jet')
#    Features.append('DER_deltar_tau_lep')
    Features.append('DER_pt_tot')
    Features.append('DER_sum_pt')
#    Features.append('DER_pt_ratio_lep_tau')
#    Features.append('DER_met_phi_centrality')
#    Features.append('DER_lep_eta_centrality')
    Features.append('PRI_tau_pt')
#    Features.append('PRI_tau_eta')
#    Features.append('PRI_tau_phi')
    Features.append('PRI_lep_pt')
#    Features.append('PRI_lep_eta')
#    Features.append('PRI_lep_phi')
    Features.append('PRI_met')
#    Features.append('PRI_met_phi')
#    Features.append('PRI_met_sumet')
#    Features.append('PRI_jet_num')
#    Features.append('PRI_jet_leading_pt')
#    Features.append('PRI_jet_leading_eta')
#    Features.append('PRI_jet_leading_phi')
#    Features.append('PRI_jet_subleading_pt')
#    Features.append('PRI_jet_subleading_eta')
#    Features.append('PRI_jet_subleading_phi')
#    Features.append('PRI_jet_all_pt')
#    Features.append('Weight')
#    Features.append('Label')

# Setup the graph
    sess = tf.Session()
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    print ("Will reduce the input data sample into the feature space for analysis, extracting the lables and weights")
    FeatureSpace = PML.ExtractColumnsTensor(sess, ReducedKaggleData, columns, Features)
    Labels       = PML.ExtractColumnTensor(sess,  StringData, stringColumns, labelName)
    Weights      = PML.ExtractColumnTensor(sess,  ReducedKaggleData, columns, weightName)

    return FeatureSpace, Features, Labels, Weights





#----------------------------------------------------------------------
#
# Main entry point to the analysis; just call RunAnalysis to ensure that 
# the script maintains a clean design and uses the api and structured
# functions.
#
#----------------------------------------------------------------------
RunAnalysis()
