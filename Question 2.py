# Gabriel Jentis
# EECE 5644
# HW 2, Question 2

import numpy as np
import matplotlib.pyplot as plt
import random

# Define True Vehicle Location
ang = random.random() * 2 * np.pi
truePos = [np.cos(ang), np.sin(ang)]  # Random Point on unit circle radius 1

# Scale to be somewhere inside unit circle
truePos[0] = truePos[0] * random.random()
truePos[1] = truePos[1] * random.random()
print("The true position of the vehicle is: " + str(truePos))

# Generate Grid to make contours
delta = 0.005
x = np.arange(-2.0, 2.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)

#  Generate K reference points
def genRefPoints(k):
    ang = random.random() * 2 * np.pi
    points = np.zeros([k,2])
    for i in range(1,k+1):
        points[i-1] = [np.cos(ang + (2*i*np.pi)/k), np.sin(ang + (2*i*np.pi)/k)]
    return points


#  Get range measurement given point and reference point
def rangeMeasurement(pt, ref):
    d = np.sqrt(np.abs(pt[0]-ref[0])**2 + np.abs(pt[1]-ref[1])**2)
    noise = np.random.normal(0, 0.3)
    meas = d + noise
    while (meas < 0):
        noise = np.random.normal(0, 0.3)
        meas = d + noise
    return meas


def MAPlikelihood(measurements, candPt, refs):
    likelihood = 0
    K = len(refs)
    for k in range(K):
        likelihood += 1.0/K * (measurements[k] - np.sqrt(np.abs(candPt[0]-refs[k][0])**2 + np.abs(candPt[1]-refs[k][1])**2))**2
    likelihood += (0.3**2)/(K*(0.25**2)) * np.matmul(candPt, np.transpose(candPt))
    return likelihood

# 1 ref point
refs1 = genRefPoints(1)
truePosRangeMes = [rangeMeasurement(truePos, r) for r in refs1]
likelihoods1 = np.zeros([len(X), len(Y)])
for i in range(len(X)):
    for j in range(len(Y)):
        likelihoods1[i, j] = MAPlikelihood(truePosRangeMes, [X[i][j], Y[i][j]], refs1)

print(MAPlikelihood(truePosRangeMes, truePos, refs1))
fig, ax = plt.subplots()
CS = ax.contour(X, Y, likelihoods1, np.linspace(likelihoods1.min(), likelihoods1.max(), 30))
ax.plot(truePos[0], truePos[1], 'r+', label='True Position')
ax.plot([refs1[i][0] for i in range(len(refs1))], [refs1[i][1] for i in range(len(refs1))], 'bo', label='Reference point')
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('MAP Objective Function Contour Map: 1 Reference Point')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.legend()
plt.show()

# 2 ref points
refs2 = genRefPoints(2)
truePosRangeMes = [rangeMeasurement(truePos, r) for r in refs2]
likelihoods2 = np.zeros([len(X), len(Y)])
for i in range(len(X)):
    for j in range(len(Y)):
        likelihoods2[i, j] = MAPlikelihood(truePosRangeMes, [X[i][j], Y[i][j]], refs2)
print(MAPlikelihood(truePosRangeMes, truePos, refs2))

fig, ax = plt.subplots()
CS = ax.contour(X, Y, likelihoods2, np.linspace(likelihoods2.min(), likelihoods2.max(), 30))
ax.plot(truePos[0], truePos[1], 'r+', label='True Position')
ax.plot([refs2[i][0] for i in range(len(refs2))], [refs2[i][1] for i in range(len(refs2))], 'bo', label='Reference point')
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('MAP Objective Function Contour Map: 2 Reference Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.legend()
plt.show()

# 3 ref points
refs3 = genRefPoints(3)
truePosRangeMes = [rangeMeasurement(truePos, r) for r in refs3]
likelihoods3 = np.zeros([len(X), len(Y)])
for i in range(len(X)):
    for j in range(len(Y)):
        likelihoods3[i, j] = MAPlikelihood(truePosRangeMes, [X[i][j], Y[i][j]], refs3)

print(MAPlikelihood(truePosRangeMes, truePos, refs3))
fig, ax = plt.subplots()
CS = ax.contour(X, Y, likelihoods3, np.linspace(likelihoods3.min(), likelihoods3.max(), 30))
ax.plot(truePos[0], truePos[1], 'r+', label='True Position')
ax.plot([refs3[i][0] for i in range(len(refs3))], [refs3[i][1] for i in range(len(refs3))], 'bo', label='Reference point')
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('MAP Objective Function Contour Map: 3 Reference Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.legend()
plt.show()

# 4 ref points
refs4 = genRefPoints(4)
truePosRangeMes = [rangeMeasurement(truePos, r) for r in refs4]
likelihoods4 = np.zeros([len(X), len(Y)])
for i in range(len(X)):
    for j in range(len(Y)):
        likelihoods4[i, j] = MAPlikelihood(truePosRangeMes, [X[i][j], Y[i][j]], refs4)

print(MAPlikelihood(truePosRangeMes, truePos, refs4))
fig, ax = plt.subplots()
CS = ax.contour(X, Y, likelihoods4, np.linspace(likelihoods4.min(), likelihoods4.max(), 30))
ax.plot(truePos[0], truePos[1], 'r+', label='True Position')
ax.plot([refs4[i][0] for i in range(len(refs4))], [refs4[i][1] for i in range(len(refs4))], 'bo', label='Reference point')
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('MAP Objective Function Contour Map: 4 Reference Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.legend()
plt.show()

# Plot Contours again, but using levels of 4 reference points
fig, ax = plt.subplots()
CS = ax.contour(X, Y, likelihoods1, np.linspace(likelihoods4.min(), likelihoods4.max(), 30))
ax.plot(truePos[0], truePos[1], 'r+', label='True Position')
ax.plot([refs1[i][0] for i in range(len(refs1))], [refs1[i][1] for i in range(len(refs1))], 'bo', label='Reference point')
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('MAP Objective Function Contour Map: 1 Reference Point')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.legend()
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(X, Y, likelihoods2, np.linspace(likelihoods4.min(), likelihoods4.max(), 30))
ax.plot(truePos[0], truePos[1], 'r+', label='True Position')
ax.plot([refs2[i][0] for i in range(len(refs2))], [refs2[i][1] for i in range(len(refs2))], 'bo', label='Reference point')
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('MAP Objective Function Contour Map: 2 Reference Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.legend()
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(X, Y, likelihoods3, np.linspace(likelihoods4.min(), likelihoods4.max(), 30))
ax.plot(truePos[0], truePos[1], 'r+', label='True Position')
ax.plot([refs3[i][0] for i in range(len(refs3))], [refs3[i][1] for i in range(len(refs3))], 'bo', label='Reference point')
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('MAP Objective Function Contour Map: 3 Reference Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.legend()
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(X, Y, likelihoods4, np.linspace(likelihoods4.min(), likelihoods4.max(), 30))
ax.plot(truePos[0], truePos[1], 'r+', label='True Position')
ax.plot([refs4[i][0] for i in range(len(refs4))], [refs4[i][1] for i in range(len(refs4))], 'bo', label='Reference point')
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('MAP Objective Function Contour Map: 4 Reference Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.legend()
plt.show()