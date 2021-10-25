# Gabriel Jentis
# EECE 5644
# HW 2, Question 1

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import scipy.optimize as spo
import pandas as pd

# Mean vectors And covariance matrices based on problem
mu01 = [5, 0]
c01 = [[4, 0],
       [0, 2]]
mu02 = [0, 4]
c02 = [[1, 0],
       [0, 3]]
mu1 = [3, 2]
c1 = [[2, 0],
      [0, 2]]

# priors based on problem
p0 = 0.6
p1 = 0.4


# This function takes in the number of samples to be generated, and generates the samples and labels based upon the
# given PDFs in the problem
def generateSamples(numSamples):
    # Generate Samples
    labels = []
    samples = []
    for n in range(0, numSamples):
        val = random.random()
        if val < p0:
            labels.append(0)
            if val < p0 / 2:
                samples.append(np.random.multivariate_normal(mu01, c01))
            else:
                samples.append(np.random.multivariate_normal(mu02, c02))
        else:
            labels.append(1)
            samples.append(np.random.multivariate_normal(mu1, c1))

    return labels, samples


# Generate samples for datasets
D_100_trainLabels, D_100_trainSamples = generateSamples(100)
D_1000_trainLabels, D_1000_trainSamples = generateSamples(1000)
D_10000_trainLabels, D_10000_trainSamples = generateSamples(10000)
D_20K_validateLabels, D_20K_validateSamples = generateSamples(20000)

# Plot data from 20K dataset to visualize, use 20K as it is biggest
samp0_20k = [x for (i, x) in enumerate(D_20K_validateSamples) if D_20K_validateLabels[i] == 0]
samp1_20k = [x for (i, x) in enumerate(D_20K_validateSamples) if D_20K_validateLabels[i] == 1]
plt.plot([x[0] for x in samp0_20k], [x[1] for x in samp0_20k], '.', color='blue', label='Class 0')
plt.plot([x[0] for x in samp1_20k], [x[1] for x in samp1_20k], '.', color='red', label='Class 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Class 0 and Class 1 True Class Labels')
plt.legend()
plt.show()

# PART 1
# Generate Discriminant scores
discrim = []

for s in D_20K_validateSamples:
    disc = multivariate_normal.pdf(s, mu1, c1) /\
           (0.5 * multivariate_normal.pdf(s, mu01, c01) + 0.5 * multivariate_normal.pdf(s, mu02, c02))
    discrim.append(disc)

labelDiscrims20k = pd.DataFrame([D_20K_validateLabels, discrim])
labelDiscrims20k = labelDiscrims20k.transpose()
labelDiscrims20k.columns = ['labels', 'discrims']
# Create Gamma thresholds for ROC curve
sortedDisc = sorted(discrim)
gammaVals = [0]
for i, d in enumerate(sortedDisc[0:-1]):
    gammaVals.append((sortedDisc[i]+sortedDisc[i+1])/2.0)

gammaVals.append(sortedDisc[-1] + 1)    # Add a gamma threshold greater than all descriminants
gammas = sorted(gammaVals)

# Generate False and true positive rates
class0Count = len(samp0_20k)
class1Count = len(samp1_20k)
falsePosRate = []
truePosRate = []
perError = []

# discrim is ordered same as samples so by looping through discrim, each sample can be observed for each gamma threshold
# value.
numGammas = len(gammas)
for ind in range(numGammas):
    trueNegCount = \
        labelDiscrims20k[(labelDiscrims20k['labels'] == 0) & (labelDiscrims20k['discrims'] < gammas[ind])].shape[0]
    falseNegCount = \
        labelDiscrims20k[(labelDiscrims20k['labels'] == 1) & (labelDiscrims20k['discrims'] < gammas[ind])].shape[0]
    falsePosCount = \
        labelDiscrims20k[(labelDiscrims20k['labels'] == 0) & (labelDiscrims20k['discrims'] > gammas[ind])].shape[0]
    truePosCount = \
        labelDiscrims20k[(labelDiscrims20k['labels'] == 1) & (labelDiscrims20k['discrims'] > gammas[ind])].shape[0]
    falsePosRate.append(falsePosCount/class0Count)
    truePosRate.append(truePosCount/class1Count)
    perError.append((falsePosCount + falseNegCount)/20000.0)

# Find minimum error for gamma threshold and which index it is
minError = min(perError)
indMinError = perError.index(minError)
print("The gamma resulting in the minimum error is: " + str(gammas[indMinError]))
print("The minimum probability of error is: " + str(minError))

# find theoretical false positive rate, true positive rate, and error
theoDisc = p0/p1
theoTrueNegCount = \
    labelDiscrims20k[(labelDiscrims20k['labels'] == 0) & (labelDiscrims20k['discrims'] < theoDisc)].shape[0]
theoFalseNegCount = \
    labelDiscrims20k[(labelDiscrims20k['labels'] == 1) & (labelDiscrims20k['discrims'] < theoDisc)].shape[0]
theoFalsePosCount = \
    labelDiscrims20k[(labelDiscrims20k['labels'] == 0) & (labelDiscrims20k['discrims'] > theoDisc)].shape[0]
theoTruePosCount = \
    labelDiscrims20k[(labelDiscrims20k['labels'] == 1) & (labelDiscrims20k['discrims'] > theoDisc)].shape[0]
theoFalsePosRate = theoFalsePosCount/class0Count
theoTruePosRate = theoTruePosCount/class1Count
theoError = (theoFalsePosCount + theoFalseNegCount)/20000.0
print("The theoretical optimal gamma is: " + str(theoDisc))
print("The theoretical minimum probability of error is: " + str(theoError))

# Plot ROC curve
plt.plot(falsePosRate, truePosRate, label='ROC curve')
plt.plot(falsePosRate[indMinError], truePosRate[indMinError], 'ro', label='Experimental Minimum Error')
plt.plot(theoFalsePosRate, theoTruePosRate, 'go', label='Theoretical Minimum Error')
plt.ylabel('P(True_Positive)')
plt.xlabel('P(False_Positive)')
plt.title('Minimum Expected Risk ROC Curve')
plt.legend()
plt.show()

# Part 2
# Get Sample Pdfs from data
# 10000 training samples
print("\n10k training set data")
samp0_10k = [x for (i, x) in enumerate(D_10000_trainSamples) if D_10000_trainLabels[i] == 0]
samp1_10k = [x for (i, x) in enumerate(D_10000_trainSamples) if D_10000_trainLabels[i] == 1]
gm_10k = GaussianMixture(n_components=2, n_init=5).fit(samp0_10k)
print("Class 0 Estimated Gaussian Weights: " + str(gm_10k.weights_))
print("Class 0 Estimated Means: " + str(gm_10k.means_))
print("Class 0 Estimated Covariances: " + str(gm_10k.covariances_))
m1Hat_10k = np.mean(samp1_10k, axis=0)
print("Class 1 Estimated Mean: " + str(m1Hat_10k))
cov1Hat_10k = np.cov(samp1_10k, rowvar=False)
print("Class 1 Estimated Covariance: " + str(cov1Hat_10k))

# 1000 training samples
print("\n1k training set data")
samp0_1k = [x for (i, x) in enumerate(D_1000_trainSamples) if D_1000_trainLabels[i] == 0]
samp1_1k = [x for (i, x) in enumerate(D_1000_trainSamples) if D_1000_trainLabels[i] == 1]
gm_1k = GaussianMixture(n_components=2, n_init=5).fit(samp0_1k)
print("Class 0 Estimated Gaussian Weights: " + str(gm_1k.weights_))
print("Class 0 Estimated Means: " + str(gm_1k.means_))
print("Class 0 Estimated Covariances: " + str(gm_1k.covariances_))
m1Hat_1k = np.mean(samp1_1k, axis=0)
print("Class 1 Estimated Mean: " + str(m1Hat_1k))
cov1Hat_1k = np.cov(samp1_1k, rowvar=False)
print("Class 1 Estimated Covariance: " + str(cov1Hat_1k))

# 100 training samples
print("\n100 training set data")
samp0_100 = [x for (i, x) in enumerate(D_100_trainSamples) if D_100_trainLabels[i] == 0]
samp1_100 = [x for (i, x) in enumerate(D_100_trainSamples) if D_100_trainLabels[i] == 1]
gm_100 = GaussianMixture(n_components=2, n_init=5).fit(samp0_100)
print("Class 0 Estimated Gaussian Weights: " + str(gm_100.weights_))
print("Class 0 Estimated Means: " + str(gm_100.means_))
print("Class 0 Estimated Covariances: " + str(gm_100.covariances_))
m1Hat_100 = np.mean(samp1_100, axis=0)
print("Class 1 Estimated Mean: " + str(m1Hat_100))
cov1Hat_100 = np.cov(samp1_10k, rowvar=False)
print("Class 1 Estimated Covariance: " + str(cov1Hat_100))

# Generate Discriminant scores
discrim10k = np.zeros((len(D_20K_validateSamples)))
discrim1k = np.zeros((len(D_20K_validateSamples)))
discrim100 = np.zeros((len(D_20K_validateSamples)))
print('a')
for i, s in enumerate(D_20K_validateSamples):
    disc10k = multivariate_normal.pdf(s, m1Hat_10k, cov1Hat_10k) / \
              (gm_10k.weights_[0] * multivariate_normal.pdf(s, gm_10k.means_[0], gm_10k.covariances_[0]) +
               gm_10k.weights_[1] * multivariate_normal.pdf(s, gm_10k.means_[1], gm_10k.covariances_[1]))
    disc1k = multivariate_normal.pdf(s, m1Hat_1k, cov1Hat_1k) / \
             (gm_1k.weights_[0] * multivariate_normal.pdf(s, gm_1k.means_[0], gm_1k.covariances_[0]) +
              gm_1k.weights_[1] * multivariate_normal.pdf(s, gm_1k.means_[1], gm_1k.covariances_[1]))
    disc100 = multivariate_normal.pdf(s, m1Hat_100, cov1Hat_100) / \
              (gm_100.weights_[0] * multivariate_normal.pdf(s, gm_100.means_[0], gm_100.covariances_[0]) +
               gm_100.weights_[1] * multivariate_normal.pdf(s, gm_100.means_[1], gm_100.covariances_[1]))
    discrim10k[i] = disc10k
    discrim1k[i] = disc1k
    discrim100[i] = disc100

labelDiscrims = pd.DataFrame([D_20K_validateLabels, discrim10k, discrim1k, discrim100])
labelDiscrims = labelDiscrims.transpose()
labelDiscrims.columns = ['labels', 'discrim10k', 'discrim1k', 'discrim100']
discrims = [discrim10k, discrim1k, discrim100]
print('b')
# Create Gamma thresholds for ROC curve
sortedDisc10k = sorted(discrim10k)
sortedDisc1k = sorted(discrim1k)
sortedDisc100 = sorted(discrim100)
sortedDiscs = [sortedDisc10k, sortedDisc1k, sortedDisc100]
print('c')
gammaVals10k = np.zeros((len(D_20K_validateSamples)) + 1)
gammaVals1k = np.zeros((len(D_20K_validateSamples)) + 1)
gammaVals100 = np.zeros((len(D_20K_validateSamples)) + 1)
for i in range(len(D_20K_validateSamples) - 1):
    gammaVals10k[i + 1] = (sortedDisc10k[i] + sortedDisc10k[i + 1]) / 2.0
    gammaVals1k[i + 1] = (sortedDisc1k[i] + sortedDisc1k[i + 1]) / 2.0
    gammaVals100[i + 1] = (sortedDisc100[i] + sortedDisc100[i + 1]) / 2.0

gammaVals10k[-1] = sortedDisc10k[-1] + 1  # Add a gamma threshold greater than all descriminants
gammaVals1k[-1] = sortedDisc1k[-1] + 1  # Add a gamma threshold greater than all descriminants
gammaVals100[-1] = sortedDisc100[-1] + 1  # Add a gamma threshold greater than all descriminants
gammas = [sorted(gammaVals10k), sorted(gammaVals1k), sorted(gammaVals100)]
print('d')
# Generate False and true positive rates
class0Count = len(samp0_20k)
class1Count = len(samp1_20k)
falsePosRate10k = []
truePosRate10k = []
perError10k = []
falsePosRate1k = []
truePosRate1k = []
perError1k = []
falsePosRate100 = []
truePosRate100 = []
perError100 = []

# discrim is ordered same as samples so by looping through discrim, each sample can be observed for each gamma threshold
# value.
numGammas = len(gammas[0])
for ind in range(numGammas):
    if ind % 1000 == 0:     # Progress of algorithm
        print(ind)
    trueNegCount10k = \
        labelDiscrims[(labelDiscrims['labels'] == 0) & (labelDiscrims['discrim10k'] < gammas[0][ind])].shape[0]
    falseNegCount10k = \
        labelDiscrims[(labelDiscrims['labels'] == 1) & (labelDiscrims['discrim10k'] < gammas[0][ind])].shape[0]
    falsePosCount10k = \
        labelDiscrims[(labelDiscrims['labels'] == 0) & (labelDiscrims['discrim10k'] > gammas[0][ind])].shape[0]
    truePosCount10k = \
        labelDiscrims[(labelDiscrims['labels'] == 1) & (labelDiscrims['discrim10k'] > gammas[0][ind])].shape[0]
    trueNegCount1k = \
        labelDiscrims[(labelDiscrims['labels'] == 0) & (labelDiscrims['discrim1k'] < gammas[1][ind])].shape[0]
    falseNegCount1k = \
        labelDiscrims[(labelDiscrims['labels'] == 1) & (labelDiscrims['discrim1k'] < gammas[1][ind])].shape[0]
    falsePosCount1k = \
        labelDiscrims[(labelDiscrims['labels'] == 0) & (labelDiscrims['discrim1k'] > gammas[1][ind])].shape[0]
    truePosCount1k = \
        labelDiscrims[(labelDiscrims['labels'] == 1) & (labelDiscrims['discrim1k'] > gammas[1][ind])].shape[0]
    trueNegCount100 = \
        labelDiscrims[(labelDiscrims['labels'] == 0) & (labelDiscrims['discrim100'] < gammas[2][ind])].shape[0]
    falseNegCount100 = \
        labelDiscrims[(labelDiscrims['labels'] == 1) & (labelDiscrims['discrim100'] < gammas[2][ind])].shape[0]
    falsePosCount100 = \
        labelDiscrims[(labelDiscrims['labels'] == 0) & (labelDiscrims['discrim100'] > gammas[2][ind])].shape[0]
    truePosCount100 = \
        labelDiscrims[(labelDiscrims['labels'] == 1) & (labelDiscrims['discrim100'] > gammas[2][ind])].shape[0]
    falsePosRate10k.append(falsePosCount10k / class0Count)
    truePosRate10k.append(truePosCount10k / class1Count)
    perError10k.append((falsePosCount10k + falseNegCount10k) / 20000.0)
    falsePosRate1k.append(falsePosCount1k / class0Count)
    truePosRate1k.append(truePosCount1k / class1Count)
    perError1k.append((falsePosCount1k + falseNegCount1k) / 20000.0)
    falsePosRate100.append(falsePosCount100 / class0Count)
    truePosRate100.append(truePosCount100 / class1Count)
    perError100.append((falsePosCount100 + falseNegCount100) / 20000.0)

print('e')
# Find minimum error for gamma threshold and which index it is
minError10k = min(perError10k)
indMinError10k = perError10k.index(minError10k)
print("The gamma resulting in the minimum error with 10000 training samples is: " + str(gammas[0][indMinError10k]))
print("The minimum probability of error with 10000 training samples is: " + str(minError10k))
minError1k = min(perError1k)
indMinError1k = perError1k.index(minError1k)
print("The gamma resulting in the minimum error with 1000 training samples is: " + str(gammas[1][indMinError1k]))
print("The minimum probability of error with 1000 training samples is: " + str(minError1k))
minError100 = min(perError100)
indMinError100 = perError100.index(minError100)
print("The gamma resulting in the minimum error with 100 training samples is: " + str(gammas[2][indMinError100]))
print("The minimum probability of error with 100 training samples is: " + str(minError100))

# Plot ROC curve separately
plt.plot(falsePosRate10k, truePosRate10k, 'b', label='ROC curve 10k Training Samples')
plt.plot(falsePosRate10k[indMinError10k], truePosRate10k[indMinError10k], 'ro',
         label='Experimental Minimum Error 10k Samples')
plt.ylabel('P(True_Positive)')
plt.xlabel('P(False_Positive)')
plt.title('Minimum Expected Risk ROC Curve 10k training samples')
plt.legend()
plt.show()

plt.plot(falsePosRate1k, truePosRate1k, 'b', label='ROC curve 1k Training Samples')
plt.plot(falsePosRate1k[indMinError1k], truePosRate1k[indMinError1k], 'ro',
         label='Experimental Minimum Error 1k Samples')
plt.ylabel('P(True_Positive)')
plt.xlabel('P(False_Positive)')
plt.title('Minimum Expected Risk ROC Curve 1k training samples')
plt.legend()
plt.show()

plt.plot(falsePosRate100, truePosRate100, 'b', label='ROC curve 100 Training Samples')
plt.plot(falsePosRate100[indMinError100], truePosRate100[indMinError100], 'ro',
         label='Experimental Minimum Error 100 Samples')
plt.ylabel('P(True_Positive)')
plt.xlabel('P(False_Positive)')
plt.title('Minimum Expected Risk ROC Curve 100 training samples')
plt.legend()
plt.show()

# Plot ROC curves Together
plt.plot(falsePosRate10k, truePosRate10k, 'b', label='ROC curve 10k Training Samples')
plt.plot(falsePosRate10k[indMinError10k], truePosRate10k[indMinError10k], 'ro',
         label='Experimental Minimum Error 10k Samples')
plt.plot(falsePosRate1k, truePosRate1k, 'c', label='ROC curve 1k Training Samples')
plt.plot(falsePosRate1k[indMinError1k], truePosRate1k[indMinError1k], 'r+',
         label='Experimental Minimum Error 1k Samples')
plt.plot(falsePosRate100, truePosRate100, 'g', label='ROC curve 100 Training Samples')
plt.plot(falsePosRate100[indMinError100], truePosRate100[indMinError100], 'r*',
         label='Experimental Minimum Error 100 Samples')
plt.ylabel('P(True_Positive)')
plt.xlabel('P(False_Positive)')
plt.title('All Training set Minimum Expected Risk ROC Curves')
plt.legend()
plt.show()

# Part 3

# Function to determine average negative log liklehood of class posteriors given sample data
# Param modelParamLin: Vector to fit sample to (ie w in 1/(1+e^wz))
# Param trainData: Set of Training Data
# Param trainLabels : Labels for training data
# Param fit : Type of fit (i.e. linear or quadratic)
# Return Average negative log likelihood of choosing correct class given sample
def logisticFunctionClassificationLikelihood(modelParam, trainData, trainLablels, fit):
    # If statement generates b(x) for each sample
    if fit == 'linear':
        z = [np.r_[1, samp] for samp in trainData]
    elif fit == 'quadratic':
        z = [np.r_[1, samp, samp[0] ** 2, samp[0] * samp[1], samp[1] ** 2] for samp in trainData]
    else:
        print('Unknown fit type for logistic classification')
        exit(-1)
        return
    # Logistic values are 1/(1+e^wz), where w is modelParamLin input and z is as calculated above
    logVals = [1.0/(1 + np.exp(np.matmul(modelParam, z[samp]))) for samp in range(len(trainData))]
    # Likelihood is 1 - logVal if label is 0
    correctLiklihood = [(1-logVals[i]) if trainLablels[i] == 0 else logVals[i] for i in range(len(trainData))]
    # Return negative mean of log likelihoods4
    return -1 * np.mean(np.log(correctLiklihood))

# Function to perform logistic-based binary classification and return the model parameters
# Param trainData: Set of Training Data
# Param trainLabels : Labels for training data
# Param initParams : initial values for the model parameters
# Param fit : Type of fit (i.e. linear or quadratic)
# Return Minimized classification function
def optimizeLogisticClassification(trainData, trainLabels, initParams, fit):
    optimizeResult = spo.minimize(fun=logisticFunctionClassificationLikelihood, x0=initParams,
                                  args=(trainData, trainLabels, fit), method='Nelder-Mead',
                                  options={'maxiter':5000, 'fatol':0.001})

    if not optimizeResult.success:
        print(optimizeResult.message)
        exit(-1)
    return optimizeResult.x

# Function Calculates minimum error probability on validation set using given parameters. Also plots the validation
# set data separated into correct and incorrect classifications
# Param params: model parameters generated form optimizing model on training data
# Param fit : fit type (i.e. linear or quadratic)
# Param info : info for labeling plots and print statements, defaults to empty
def plotLogClassPerformance(params, fit, info=''):
    # Calculate minimum error probabilities on validation set based on fit type and parameters
    if fit == 'linear':
        likelihoods = [params[0] + params[1] * D_20K_validateSamples[i][0] +
                       params[2] * D_20K_validateSamples[i][1] for i in range(20000)]
    elif fit == 'quadratic':
        likelihoods = [params[0] + params[1] * D_20K_validateSamples[i][0] +
                       params[2] * D_20K_validateSamples[i][1] +
                       params[3] * (D_20K_validateSamples[i][0] ** 2) +
                       params[4] * D_20K_validateSamples[i][0] * D_20K_validateSamples[i][1] +
                       params[5] * (D_20K_validateSamples[i][1] ** 2) for i in range(20000)]
    else:
        print('Unknown Fit Type')
        exit(-1)
        return

    decisions = [int(i < 0.5) for i in likelihoods]
    numErrors = 0
    for i in range(20000):
        if decisions[i] != D_20K_validateLabels[i]:
            numErrors += 1
    errorProb = numErrors/20000.0
    print('Probability of error ' + info +': ' + str(errorProb))

    # Plot Data As Classified Correct or incorrect
    class0Correct = [s for (i, s) in enumerate(D_20K_validateSamples) if D_20K_validateLabels[i] == 0 and
                     decisions[i] == 0]
    class0Incorrect = [s for (i, s) in enumerate(D_20K_validateSamples) if D_20K_validateLabels[i] == 0 and
                       decisions[i] == 1]
    class1Correct = [s for (i, s) in enumerate(D_20K_validateSamples) if D_20K_validateLabels[i] == 1 and
                     decisions[i] == 1]
    class1Incorrect = [s for (i, s) in enumerate(D_20K_validateSamples) if D_20K_validateLabels[i] == 1 and
                       decisions[i] == 0]

    plt.plot([x[0] for x in class0Correct], [x[1] for x in class0Correct], '+', color='green',
             label='Class 0 Correct')
    plt.plot([x[0] for x in class0Incorrect], [x[1] for x in class0Incorrect], '+', color='red',
             label='Class 0 Incorrect')
    plt.plot([x[0] for x in class1Correct], [x[1] for x in class1Correct], 'x', color='green',
             label='Class 1 Correct')
    plt.plot([x[0] for x in class1Incorrect], [x[1] for x in class1Incorrect], 'x', color='red',
             label='Class 1 Incorrect')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Logistic ' + fit + ' Classification ' + info)
    plt.legend()
    plt.show()


# Perform logistic linear classification
modelParamLin = np.array([0, 0, 0])
modelParamLin = optimizeLogisticClassification(D_100_trainSamples, D_100_trainLabels, modelParamLin, 'linear')
print(modelParamLin)
plotLogClassPerformance(modelParamLin, 'linear', '100 Train Samples')
modelParamLin = optimizeLogisticClassification(D_1000_trainSamples, D_1000_trainLabels, modelParamLin, 'linear')
print(modelParamLin)
plotLogClassPerformance(modelParamLin, 'linear', '1k Train Samples')
modelParamLin = optimizeLogisticClassification(D_10000_trainSamples, D_10000_trainLabels, modelParamLin, 'linear')
print(modelParamLin)
plotLogClassPerformance(modelParamLin, 'linear', '10k Train Samples')

# Perform logistic quadratic classification
modelParamQuad = np.array([0, 0, 0, 0, 0, 0])
modelParamQuad = optimizeLogisticClassification(D_100_trainSamples, D_100_trainLabels, modelParamQuad, 'quadratic')
print(modelParamQuad)
plotLogClassPerformance(modelParamQuad, 'quadratic', '100 Train Samples')
modelParamQuad = optimizeLogisticClassification(D_1000_trainSamples, D_1000_trainLabels, modelParamQuad, 'quadratic')
print(modelParamQuad)
plotLogClassPerformance(modelParamQuad, 'quadratic', '1k Train Samples')
modelParamQuad = optimizeLogisticClassification(D_10000_trainSamples, D_10000_trainLabels, modelParamQuad, 'quadratic')
print(modelParamQuad)
plotLogClassPerformance(modelParamQuad, 'quadratic', '10k Train Samples')
