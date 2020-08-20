# %% Imports
import re
import numpy as np
import random
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

sys.path.insert(1, "../utils")

from separateIndexesByRatio import separateIndexesByRatio


# %% Declare functions
def getFileData(path, cols):
    data = np.loadtxt(path, usecols=cols)
    return data


def getFileDataAlt(path):
    data = open(path, "r").read()
    data = re.sub(r'"\w*"\n?|| ', "", data)
    data = data.split("\n")
    data.pop()
    data = [float(val) for val in data]
    return data


def addConstantTerm(arr):
    oneVector = np.ones((arr.shape[0], 1))
    newArr = np.concatenate((arr, oneVector), axis=1)
    return newArr


class Adaline:
    def __init__(self, weights=[1], learnRate=0.1):
        self.__weights = np.array(weights)
        self.__learnRate = learnRate
        self.__nTrains = 0

    def getWeights(self):
        return self.__weights

    def evaluate(self, xArr):
        approxY = np.dot(xArr, np.transpose(self.__weights))
        return approxY

    def singleTrain(self, xArr, y):
        approxY = self.evaluate(xArr)
        error = y - approxY
        self.__weights = np.add(self.__weights, self.__learnRate * error * xArr)
        self.__nTrains += 1
        return error

    def test(self, xArr, y):
        approxY = self.evaluate(xArr)
        return y - approxY

    def train(self, xMatrix, yArr, tol=1e-5, maxIterations=1):
        for _ in range(maxIterations):
            iterationError = 0
            for index in range(len(yArr)):
                error = adaline.singleTrain(xMatrix[index], yArr[index])
                iterationError += error ** 2

            if iterationError < tol:
                break

    def printDetails(self):
        print(f"weights: {self.__weights}\nTimes trained: {self.__nTrains}")


# %% Initialize data
filePrefix = "Ex1"
xDimension = 1
tol = 1e-14
maxIterations = 100

timeSamples = getFileData(f"data/{filePrefix}_t", (1))
xSamples = np.array(
    getFileData(f"data/{filePrefix}_x", tuple(range(1, xDimension + 1)))
)

if xDimension == 1:
    xSamples = xSamples.reshape(len(xSamples), 1)

adalineXSamples = addConstantTerm(xSamples)
ySamples = getFileData(f"data/{filePrefix}_y", (1))

trainSamplesRatio = 0.7
trainIndexes, testIndexes = separateIndexesByRatio(len(timeSamples), 0.7)

# %% Initialize and Train Adaline
adaline = Adaline([1] * (xDimension + 1), 0.1)

xTrain = adalineXSamples[trainIndexes]
yTrain = ySamples[trainIndexes]
adaline.train(xTrain, yTrain, tol, maxIterations)

# %% Test
testResult = np.array([])
for index in testIndexes:
    testResult = np.append(
        testResult, adaline.test(adalineXSamples[index], ySamples[index])
    )

squarer = np.vectorize(lambda x: x ** 2)
meanSquaredError = np.mean(squarer(testResult))
print(meanSquaredError)

# %%
adalineResult = np.array([])
for x in adalineXSamples:
    adalineResult = np.append(adalineResult, adaline.evaluate(x))


# %%
fig = make_subplots()

for xSample in xSamples.T:
    fig.add_trace(go.Scatter(x=timeSamples, y=xSample, name="Entrada"))

fig.add_trace(go.Scatter(x=timeSamples, y=ySamples, name="Saida"))

fig.add_trace(go.Scatter(x=timeSamples, y=adalineResult, name="Adaline"))

fig.show()

# %%
adaline.printDetails()
