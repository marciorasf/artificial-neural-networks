# %%
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


# %%
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


# %%
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
        learnResult = self.__learnRate * (y - approxY)
        self.__weights = np.add(self.__weights, learnResult * xArr)
        self.__nTrains += 1

    def test(self, xArr, y):
        approxY = self.evaluate(xArr)
        return y - approxY

    def train(self, xMatrix, yArr, tol=1e-5, maxIterations=1):
        for _ in range(maxIterations):
            for index in range(len(yArr)):
                adaline.singleTrain(xMatrix[index], yArr[index])

    def printDetails(self):
        print(f"weights: {self.__weights}\nTimes tested: {self.__nTrains}")


# %% Initialize data
timeSamples = getFileData("data/Ex2_t", (1))
xSamples = getFileData("data/Ex2_x", (1, 2, 3))
adalineXSamples = addConstantTerm(xSamples)
ySamples = getFileData("data/Ex2_y", (1))

trainSamplesRatio = 0.7
trainIndexes, testIndexes = separateIndexesByRatio(len(timeSamples), 0.7)

# %% Initialize and Train Adaline
adaline = Adaline([1, 1, 1, 1], 0.1)

xTrain = adalineXSamples[trainIndexes]
yTrain = ySamples[trainIndexes]
adaline.train(xTrain, yTrain, 1e-5, 10)
# %% Test
testResult = np.array([])
for index in testIndexes:
    testResult = np.append(
        testResult, adaline.test(adalineXSamples[index], ySamples[index])
    )

squarer = np.vectorize(lambda xSample: xSample ** 2)
meanSquaredError = np.mean(squarer(testResult))
print(meanSquaredError)


# %%
adalineResult = np.array([])
for sampleInput in adalineXSamples:
    adalineResult = np.append(adalineResult, adaline.evaluate(sampleInput))


# %%
fig = make_subplots()

# for xSample in xSamples.T:
#     fig.add_trace(go.Scatter(x=timeSamples, y=xSample, name="Entrada"))

fig.add_trace(go.Scatter(x=timeSamples, y=ySamples, name="Saida"))

fig.add_trace(go.Scatter(x=timeSamples, y=adalineResult, name="Adaline"))

fig.show()

# %%
adaline.printDetails()
