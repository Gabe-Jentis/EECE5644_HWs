# Gabriel Jentis
# EECE 5644
# HW 3, Question 1

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.mixture import GaussianMixture

# Data and priors
p0 = 0.3
m0 = [4, 4]
c0 = [[1, 0.4],
      [0.4, 1]]

p1 = 0.2
m1 = [3, -0.5]
c1 = [[1., -0.5],
      [-0.5, 1.]]

p2 = 0.15
m2 = [0.25, 0.25]
c2 = [[1, 0.5],
      [0.5, 1]]

p3 = 0.35
m3 = [1, 3]
c3 = [[0.5, -0.3],
      [-0.3, 0.5]]

def generateSamples(numSamples):
    # Generate Samples
    samples = []
    for n in range(0, numSamples):
        val = random.random()
        if val < p0:
            samples.append(np.random.multivariate_normal(m0, c0))
        elif val < p0 + p1:
            samples.append(np.random.multivariate_normal(m1, c1))
        elif val < p0 + p1 + p2:
            samples.append(np.random.multivariate_normal(m2, c2))
        else:
            samples.append(np.random.multivariate_normal(m3, c3))
    return np.array(samples)


# Create Data Sets
data10 = generateSamples(10)
data100 = generateSamples(100)
data1000 = generateSamples(1000)
data10000 = generateSamples(10000)

# Plot Datasets
plt.plot(data10[:, 0], data10[:, 1], 'b.', label='Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('10 Sample GMM Dataset')
plt.legend()
plt.show()

plt.plot(data100[:, 0], data100[:, 1], 'b.', label='Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('100 Sample GMM Dataset')
plt.legend()
plt.show()

plt.plot(data1000[:, 0], data1000[:, 1], 'b.', label='Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('1000 Sample GMM Dataset')
plt.legend()
plt.show()

plt.plot(data10000[:, 0], data10000[:, 1], 'b.', label='Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('10000 Sample GMM Dataset')
plt.legend()
plt.show()


# This function finds the log likelihood that a particular Gaussian Mixture model classifies validation data
# Param trainData : Dataset to fit GMM to
# Param validData : Data to find log likelihood of fitted GMM
# Param numGauss : Number of Gaussian Compnents to create GMM
# Returns log Likelihood of GMM on validation data
def getGMMlikelihood(trainData, validData, numGauss):
    gm = GaussianMixture(n_components=numGauss, n_init=15, init_params='random').fit(trainData)
    likelihood = gm.score(validData)
    return likelihood

# This function performs k-fold validation on a set of training data using a given function to serve as a metric for
# its performance for different Model orders
# Param samples : Samples of Gaussian being Classified
# Param K : Number of  parts to partition data for training and validation
# Param verbose : Flag to print progress messages
# Returns Number of Gaussians in mixture model
def kFoldCrossValidation(samples, K, verbose):
    # Get indices to partition data for K folds
    partitionInd = np.r_[np.linspace(0, samples.shape[0], num=K, endpoint=False, dtype=int), samples.shape[0]]
    # Loop through the partitions as validation data
    bestPerfOrders = np.zeros(K)

    for k in range(K):
        # Partition data into Training and Validation data for current iteration
        tempTrainData = np.r_[samples[:partitionInd[k]], samples[partitionInd[k + 1]:]]
        tempValidData = samples[partitionInd[k]:partitionInd[k + 1]]

        bestPerf = -100000000  # Very low number to initialize
        bestPerfOrder = 0
        # Increase model order until performance decreases stopConsecDec times
        for modelOrder in range(1,7):
            perf = getGMMlikelihood(tempTrainData, tempValidData, modelOrder)

            # Check if current model order was best performance
            if perf > bestPerf:
                bestPerf = perf
                bestPerfOrder = modelOrder

            # Print status if verbose asked for
            if verbose:
                print(str(modelOrder) + ' Gaussians for K ' + str(k + 1) + '/' + str(K) + ', sample size = ' +
                      str(samples.shape[0]) + ', likelihood = ' + str(perf))

            # Prep next iteration
            modelOrder += 1

        bestPerfOrders[k] = bestPerfOrder
        if verbose:
            print('K ' + str(k + 1) + '/' + str(K) + ' complete, sample size = ' + str(samples.shape[0]) +
                  ', chosen number Gaussians = ' + str(bestPerfOrder))

    returnOrder = np.mean(bestPerfOrders)
    returnOrder = int(np.round_(returnOrder, decimals=0))
    if verbose:
        print('sample size = ' + str(samples.shape[0]) + ' complete, chosen number Gaussians = ' + str(returnOrder))
    return returnOrder

# This function runs the K fold cross validation a specified number of times and returns the counts of the number of
# Gaussians the runs determined to be best for training data
# Param samples : samples for Gaussian mixture being observed
# Param K : Number of  parts to partition data for training and validation
# Param numRuns : Number of times to evaluate number of Gaussians in mixture model
# Returns Array with counts of how many times each number of Gaussians was chosen as correct
def getOrderCounts(samples, K, numRuns):
    orders = []
    for x in range(numRuns):
        currentOrd = kFoldCrossValidation(samples, K, False)
        orders.append(currentOrd)
        print('Run ' + str(x+1) + '/' + str(numRuns) + ' for ' + str(samples.shape[0]) + ' samples')

    numGauss, count = np.unique(orders, return_counts=True)
    gaussCounts = dict(zip(numGauss, count))
    counts = np.zeros(6, dtype=int)

    for i in range(1, 7):
        try:
            counts[i - 1] = gaussCounts[i]
        except KeyError:
            counts[i - 1] = 0

    return counts

# Classify for each set of sample data
K_FOLD = 10
NUM_RUNS = 30

orders10 = getOrderCounts(data10, K_FOLD, NUM_RUNS)
print('10 Samples: ', orders10)
orders100 = getOrderCounts(data100, K_FOLD, NUM_RUNS)
print('100 Samples: ', orders100)
orders1000 = getOrderCounts(data1000, K_FOLD, NUM_RUNS)
print('1000 Samples: ', orders1000)
orders10000 = getOrderCounts(data10000, K_FOLD, NUM_RUNS)
print('10000 Samples: ', orders10000)

# Plot Chosen Gaussians in Bar Graphs
# Number of gaussians array for x-axis of plots
numGaussians = [x for x in range(1,7)]

plt.bar(numGaussians, orders10, label='Number Chosen Gaussians')
plt.xlabel('Number of Gaussians in Mixture')
plt.ylabel('Number of times Chosen')
plt.title('Number of Gaussians Chosen for Mixture from 10 Samples')
plt.legend()
plt.show()

plt.bar(numGaussians, orders100, label='Number Chosen Gaussians')
plt.xlabel('Number of Gaussians in Mixture')
plt.ylabel('Number of times Chosen')
plt.title('Number of Gaussians Chosen for Mixture from 100 Samples')
plt.legend()
plt.show()

plt.bar(numGaussians, orders1000, label='Number Chosen Gaussians')
plt.xlabel('Number of Gaussians in Mixture')
plt.ylabel('Number of times Chosen')
plt.title('Number of Gaussians Chosen for Mixture from 1000 Samples')
plt.legend()
plt.show()

plt.bar(numGaussians, orders10000, label='Number Chosen Gaussians')
plt.xlabel('Number of Gaussians in Mixture')
plt.ylabel('Number of times Chosen')
plt.title('Number of Gaussians Chosen for Mixture from 10000 Samples')
plt.legend()
plt.show()