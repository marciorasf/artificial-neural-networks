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
def getFileData(path):
    data = np.loadtxt(path, usecols=(1))
    return data


def getFileDataAlt(path):
    data = open(path, "r").read()
    data = re.sub(r'"\w*"\n?|| ', "", data)
    data = data.split("\n")
    data.pop()
    data = [float(val) for val in data]
    return data


def addConstantTerm(arr):
    mapped = map(lambda x: [x, 1], arr)
    return np.array(list(mapped))


# %%
class Adaline:
    def __init__(self, weights=[1], learnRate=1):
        self.__weights = np.array(weights)
        self.__learnRate = learnRate
        self.__nTrains = 0

    def getWeights(self):
        return self.__weights

    def evaluate(self, xArr):
        approxY = np.dot(xArr, np.transpose(self.__weights))
        return approxY

    def train(self, xArr, y):
        approxY = self.evaluate(xArr)
        learnResult = self.__learnRate * (y - approxY)
        self.__weights = np.add(self.__weights, np.multiply(learnResult, xArr))
        self.__nTrains += 1

    def test(self, xArr, y):
        approxY = self.evaluate(xArr)
        return y - approxY

    def printDetails(self):
        print(f"weights: {self.__weights}\nTimes tested: {self.__nTrains}")


# %% Initialize data
timeSamples = getFileData("data/Ex1_t")
xSamples = getFileData("data/Ex1_x")
adalineXSamples = addConstantTerm(xSamples)
ySamples = getFileData("data/Ex1_y")

trainSamplesRatio = 0.7
trainIndexes, testIndexes = separateIndexesByRatio(len(adalineXSamples), 0.7)

# %% Initialize Adaline
adaline = Adaline([1, 1])

# %% Train
for index in trainIndexes:
    adaline.train(adalineXSamples[index], ySamples[index])

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


# %%
adalineResult = np.array([])
for sampleInput in adalineXSamples:
    adalineResult = np.append(adalineResult, adaline.evaluate(sampleInput))


# %%
fig = make_subplots()

fig.add_trace(go.Scatter(x=timeSamples, y=xSamples, name="Entrada"),)

fig.add_trace(go.Scatter(x=timeSamples, y=ySamples, name="Saida"),)

fig.add_trace(go.Scatter(x=timeSamples, y=adalineResult, name="Adaline"),)

fig.show()

# %%
adaline.printDetails()
