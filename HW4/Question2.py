# Gabriel Jentis
# EECE 5644
# HW 3, Question 2

from PIL import Image  # Referenced PIL documentation for how to read image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# This function takes an image and makes an array of the pixels in the unit hypercube
def createHypercube(img):
    try:
        im = Image.open(img)
        im = im.resize((161,107))
        #im = im.resize((40, 26))
    except FileNotFoundError:
        # Print error message and exit if path to image is invalid
        print('Not a valid file path to image')
        exit(1)
    imArr = np.asarray(im)
    imArrx = imArr.shape[0]
    imArry = imArr.shape[1]
    hypercube = []
    for x in range(imArrx):
        for y in range(imArry):
            # Normalize data in hypercube
            hypercube.append(np.r_[x, y, x/imArrx, y/imArry, [a/255. for a in imArr[x, y, :]]])
    hypercube = np.array(hypercube)
    print(hypercube.shape)
    return imArrx, imArry, hypercube


# This function finds the log likelihood that a particular Gaussian Mixture model classifies validation data
# Param trainData : Dataset to fit GMM to
# Param validData : Data to find log likelihood of fitted GMM
# Param numGauss : Number of Gaussian Compnents to create GMM
# Returns log Likelihood of GMM on validation data
def getGMMlikelihood(trainData, validData, numGauss):
    gm = GaussianMixture(n_components=numGauss, n_init=10, init_params='random').fit(trainData)
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
        # Increase model order
        for modelOrder in range(1,11):
            perf = getGMMlikelihood(tempTrainData, tempValidData, modelOrder)

            # Check if current model order was best performance
            if perf > bestPerf:
                bestPerf = perf
                bestPerfOrder = modelOrder

            # Print status if verbose asked for
            if verbose:
                print(str(modelOrder) + ' Gaussians for K ' + str(k + 1) + '/' + str(K) + ', sample size = ' +
                      str(samples.shape[0]) + ', likelihood = ' + str(perf))


        bestPerfOrders[k] = bestPerfOrder
        if verbose:
            print('K ' + str(k + 1) + '/' + str(K) + ' complete, sample size = ' + str(samples.shape[0]) +
                  ', chosen number Gaussians = ' + str(bestPerfOrder))

    returnOrder = np.mean(bestPerfOrders)
    returnOrder = int(np.round_(returnOrder, decimals=0))
    if verbose:
        print('sample size = ' + str(samples.shape[0]) + ' complete, chosen number Gaussians = ' + str(returnOrder))
    return returnOrder

# Get Hypercube Representation of the image
xDim, yDim, birdData = createHypercube('Bird.jpg')
np.random.shuffle(birdData)

# Run 10 fold Cross Validation to get ideal number components for model
idealClustNum = kFoldCrossValidation(birdData[:, 2:], 10, True)

# Fit all data to ideal number of clusters
model = GaussianMixture(n_components=idealClustNum, n_init=15, init_params='random').fit(birdData[:,2:])

labels = np.full(birdData.shape[0], -1)
for d in range(len(labels)):
    likelihood = np.zeros(idealClustNum)
    for c in range(idealClustNum):
        likelihood[c] = model.weights_[c] * multivariate_normal.pdf(birdData[d, 2:], model.means_[c], model.covariances_[c])
    labels[d] = np.argmax(likelihood)

segmentImg = np.full([xDim, yDim], -1.)

for i in range(len(labels)):
    segmentImg[int(birdData[i, 0])][int(birdData[i, 1])] = (1.0*labels[i])/idealClustNum

bird2 = Image.fromarray(np.uint8(segmentImg * 255), 'L')
bird2.save('segmentedImg5.png')
