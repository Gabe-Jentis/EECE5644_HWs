# Gabriel Jentis
# EECE 5644
# HW 3, Question 1

import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn.neural_network as skl
from scipy.stats import multivariate_normal

# Data
m0 = [1.5, 1, 1]
c0 = [[1, 0.3, 0.2],
      [0.3, 1, 0.62],
      [0.2, 0.62, 0.7]]

m1 = [3, 0, 2]
c1 = [[0.8, 0.47, 0.47],
      [0.47, 0.8, 0.8],
      [0.47, 0.8, 1]]

m2 = [1, 2, 2.5]
c2 = [[0.5, 0.2, 0.1],
      [0.2, 0.8, 0.5],
      [0.1, 0.5, 0.76]]

m3 = [2, 2, 0]
c3 = [[1.1, 0.25, 0.5],
      [0.25, 1.34, 0.4],
      [0.5, 0.4, 1]]
# All classes have equal priors
p = 0.25


def generateSamples(numSamples):
    # Generate Samples
    labels = []
    samples = []
    for n in range(0, numSamples):
        val = random.random()
        if val < p:
            labels.append(0)
            samples.append(np.random.multivariate_normal(m0, c0))
        elif val < 2 * p:
            labels.append(1)
            samples.append(np.random.multivariate_normal(m1, c1))
        elif val < 3 * p:
            labels.append(2)
            samples.append(np.random.multivariate_normal(m2, c2))
        else:
            labels.append(3)
            samples.append(np.random.multivariate_normal(m3, c3))
    return np.array(labels), np.array(samples)


# Generate datasets
d_train_100_labels, d_train_100 = generateSamples(100)
d_train_200_labels, d_train_200 = generateSamples(200)
d_train_500_labels, d_train_500 = generateSamples(500)
d_train_1000_labels, d_train_1000 = generateSamples(1000)
d_train_2000_labels, d_train_2000 = generateSamples(2000)
d_train_5000_labels, d_train_5000 = generateSamples(5000)
print('here')
d_test_labels, d_test = generateSamples(100000)
print('here')

#  Find Theoretically Optimal Probabilty of Error with known pdf
# Create Lambda Matrix, and organize means, covariances and priors to single arrays
lambdaMat = [[0, 1, 1, 1],
             [1, 0, 1, 1],
             [1, 1, 0, 1],
             [1, 1, 1, 0]]


# Function to find risk for a given sample, given a class guess and lamda matrix used
# Acts as risk function R(D=A|x)
def sampRisk(guessClass, samp, lambMat):
    totRisk = 0
    # Class 0
    totRisk += lambMat[guessClass][0] * p * multivariate_normal.pdf(samp, m0, c0)
    # Class 1
    totRisk += lambMat[guessClass][1] * p * multivariate_normal.pdf(samp, m1, c1)
    # Class 2
    totRisk += lambMat[guessClass][2] * p * multivariate_normal.pdf(samp, m2, c2)
    # Class3
    totRisk += lambMat[guessClass][3] * p * multivariate_normal.pdf(samp, m3, c3)
    return totRisk


numWrong = 0
for ind in range(100000):
    if ind % 1000 == 0:
        print(ind)
    s = d_test[ind]
    guess = np.argmin(
        [sampRisk(0, s, lambdaMat), sampRisk(1, s, lambdaMat), sampRisk(2, s, lambdaMat), sampRisk(3, s, lambdaMat)])
    if guess != d_test_labels[ind]:
        numWrong += 1

theoError = numWrong / 100000.
print('The theoretical optimal minimum probability of error is: ' + str(theoError))

# Plot 5000 dataset to get visualization of data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
s0 = np.array([d_train_5000[x] for x in range(5000) if d_train_5000_labels[x] == 0])
s1 = np.array([d_train_5000[x] for x in range(5000) if d_train_5000_labels[x] == 1])
s2 = np.array([d_train_5000[x] for x in range(5000) if d_train_5000_labels[x] == 2])
s3 = np.array([d_train_5000[x] for x in range(5000) if d_train_5000_labels[x] == 3])

ax.scatter(xs=s0[:, 0], ys=s0[:, 1], zs=s0[:, 2], c='b', label='Class 0')
ax.scatter(xs=s1[:, 0], ys=s1[:, 1], zs=s1[:, 2], c='g', label='Class 1')
ax.scatter(xs=s2[:, 0], ys=s2[:, 1], zs=s2[:, 2], c='r', label='Class 2')
ax.scatter(xs=s3[:, 0], ys=s3[:, 1], zs=s3[:, 2], c='y', label='Class 3')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('True Class Labels')
plt.legend()
plt.show()


# This function finds the performance data of a neural network on validation or test data
# Param nn : Neural Network being observed
# Param samples : Validation or test samples
# Param labels : Validation or test labels
# Return : Log-likelihood of correctly classifying validation/test data
def findNNPerformance(nn, samples, labels):
    predictions = nn.predict(samples)
    correctClassifications = 0
    for i in range(samples.shape[0]):
        correctClassifications += 1 if predictions[i] == labels[i] else 0
    lenData = samples.shape[0]
    return np.log(correctClassifications / lenData)


# Function to train neural network on given training data with a provided model order with 1 hidden layer of perceptrons
# it then returns the performance of the trained network on given validation data
# Param trainData : Data to train neural network on
# Param trainLabel : Labels for training data
# Param validData : Data to validate neural network with
# Param validLabel : Labels for validation data
# Param modelOrder : The number of perceptrons in a single hidden layer
# Param retNN : Boolean flag indicating whether to return the neural network
# Returns : Log Likelihood of correct classification on validation data with trained model
# If retNN is true, the neural network model is also returned
def getNNTrainPerformance(trainData, trainLabel, validData, validLabel, modelOrder, retNN):
    nn = skl.MLPClassifier(hidden_layer_sizes=(modelOrder,), activation='relu', solver='adam',
                           alpha=1e-6, max_iter=3000, shuffle=True, tol=1e-4, verbose=False,
                           warm_start=False, early_stopping=False, n_iter_no_change=10)
    # Fit network to Train data
    nn.fit(trainData, trainLabel)
    logLikelihood = findNNPerformance(nn, validData, validLabel)
    # Return logLikelihood, if retNN also return neural network
    if retNN:
        return logLikelihood, nn
    else:
        return logLikelihood


# This function performs k-fold validation on a set of training data using a given function to serve as a metric for
# its performance for different Model orders
# Param trainData : Data to train neural network on
# Param trainLabel : Labels for training data
# Param K : Number of  parts to partition data for training and validation
# Param  stopConsecDec : Number of consecutive performance decreases before stopping model order increases
# Param verbose : Flag to print progress messages
# Param initOrder : inital model order, defaults to 1
# Param orderStep : How much to increase order step each time through, defaults to 1
# Returns Model Order for training data
def kFoldCrossValidation(trainData, trainLabel, K, stopConsecDec, verbose, initOrder=1, orderStep=1):
    # Get indices to partition data for K folds
    partitionInd = np.r_[np.linspace(0, trainData.shape[0], num=K, endpoint=False, dtype=int), trainData.shape[0]]
    # Loop through the partitions as validation data
    bestPerfOrders = np.zeros(K)

    for k in range(K):
        # Partition data into Training and Validation data for current iteration
        tempTrainData = np.r_[trainData[:partitionInd[k]], trainData[partitionInd[k + 1]:]]
        tempTrainLabel = np.r_[trainLabel[:partitionInd[k]], trainLabel[partitionInd[k + 1]:]]
        tempValidData = trainData[partitionInd[k]:partitionInd[k + 1]]
        tempValidLabel = trainLabel[partitionInd[k]:partitionInd[k + 1]]

        consecPerfDecreases = 0
        lastPerf = -100000000  # Very low number to initialize
        bestPerf = -100000000  # Very low number to initialize
        bestPerfOrder = 0
        modelOrder = initOrder
        # Increase model order until performance decreases stopConsecDec times
        while consecPerfDecreases < stopConsecDec:
            perf = getNNTrainPerformance(tempTrainData, tempTrainLabel, tempValidData, tempValidLabel, modelOrder,
                                         False)

            # Check if perf worse than last performance
            if perf <= lastPerf:
                consecPerfDecreases += 1
            else:
                consecPerfDecreases = 0

            # Check if current model order was best performance
            if perf > bestPerf:
                bestPerf = perf
                bestPerfOrder = modelOrder

            # Print status if verbose asked for
            if verbose:
                print(str(modelOrder) + ' model order for K ' + str(k + 1) + '/' + str(K) + ', sample size = ' +
                      str(trainData.shape[0]) + ', performance = ' + str(perf))

            # Prep next iteration
            lastPerf = perf
            modelOrder += orderStep

        bestPerfOrders[k] = bestPerfOrder
        if verbose:
            print('K ' + str(k + 1) + '/' + str(K) + ' complete, sample size = ' + str(trainData.shape[0]) +
                  ', chosen order = ' + str(bestPerfOrder))

    returnOrder = np.mean(bestPerfOrders)
    if verbose:
        print('sample size = ' + str(trainData.shape[0]) + ' complete, chosen order = ' + str(returnOrder))
    return returnOrder


# This function finds the number of perceptrons and error probability of error for given training data set on
# given test data sets. Use 10-cross fold validation to get the number of perceptrons to use
# Param trainData : Data to train neural network on
# Param trainLabel : Labels for training data
# Param validData : Data to test neural network with
# Param validLabel : Labels for test data
# Param k : Number of  parts to partition data for training and validation
# Param numInits: Number of times to train model on number peceptrons
# Returns : Number of perceptrons in final model and error probability
def getBestTrain(trainData, trainLabel, testData, testLabel, k, numInits):
    # Use k-fold validation to get number of perceptrons with training set
    perceptrons = kFoldCrossValidation(trainData, trainLabel, k, 3, False)
    perceptrons = int(np.round_(perceptrons, decimals=0))

    # Train data using number of perceptrons from cross validation multiple times and take best train
    bestTrain = None
    bestTrainPerf = -10000000  # Very Low Number to initialize

    for i in range(numInits):
        perf, neuralNet = getNNTrainPerformance(trainData, trainLabel, trainData, trainLabel, perceptrons, True)

        if perf > bestTrainPerf:
            bestTrainPerf = perf
            bestTrain = neuralNet

    # Classify test data set with best trained nn
    logLikelihood = findNNPerformance(bestTrain, testData, testLabel)
    errorProb = 1 - np.exp(logLikelihood)
    print('Error Probability for ' + str(trainData.shape[0]) + ' samples: ' + str(errorProb))
    print('Number of Perceptrons for ' + str(trainData.shape[0]) + ' samples: ' + str(perceptrons))
    return errorProb, perceptrons


K_FOLD = 10
NUM_TRAIN_INITS = 5

errorProb100, perceptrons100 = getBestTrain(d_train_100, d_train_100_labels, d_test, d_test_labels,
                                            K_FOLD, NUM_TRAIN_INITS)
errorProb200, perceptrons200 = getBestTrain(d_train_200, d_train_200_labels, d_test, d_test_labels,
                                            K_FOLD, NUM_TRAIN_INITS)
errorProb500, perceptrons500 = getBestTrain(d_train_500, d_train_500_labels, d_test, d_test_labels,
                                            K_FOLD, NUM_TRAIN_INITS)
errorProb1000, perceptrons1000 = getBestTrain(d_train_1000, d_train_1000_labels, d_test, d_test_labels,
                                              K_FOLD, NUM_TRAIN_INITS)
errorProb2000, perceptrons2000 = getBestTrain(d_train_2000, d_train_2000_labels, d_test, d_test_labels,
                                              K_FOLD, NUM_TRAIN_INITS)
errorProb5000, perceptrons5000 = getBestTrain(d_train_5000, d_train_5000_labels, d_test, d_test_labels,
                                              K_FOLD, NUM_TRAIN_INITS)

# Plot error probability over number of samples with theoretical minimum
errorProbs = [errorProb100, errorProb200, errorProb500, errorProb1000, errorProb2000, errorProb5000]
numSamps = [100, 200, 500, 1000, 2000, 5000]

plt.plot(numSamps, errorProbs, 'b', label='Training Minimum Error')
plt.plot([numSamps[0], numSamps[-1]], [theoError] * 2, 'g--', label='Theoretical Minimum Probability Error')
plt.xscale('log')
plt.title('Minimum Classification Error on Test Data Using Different Training Sizes')
plt.xlabel('Number of Training Samples')
plt.ylabel('Minimum Classification Error on Test Data')
plt.legend()
plt.show()

# Plot number of perceptrons chosen for number of training samples
percepts = [perceptrons100, perceptrons200, perceptrons500, perceptrons1000, perceptrons2000, perceptrons5000]
plt.plot(numSamps, percepts, 'b', label='Number of Chosen Perceptrons')
plt.xscale('log')
plt.title('Number of Chosen perceptrons for Different Training Sizes')
plt.xlabel('Number of Training Samples')
plt.ylabel('Number of Chosen Perceptrons')
plt.legend()
plt.show()