# %%
import re
import numpy as np


def getFileData(path):
    data = np.loadtxt(path, usecols=range(1, 2))
    return data


def getFileDataAlt(path):
    data = open(path, "r").read()
    data = re.sub(r'"\w*"\n?|| ', "", data)
    data = data.split("\n")
    data.pop()
    data = [float(val) for val in data]
    return data


# %%
class Adaline:
    def __init__(self, weights=[1], learnRate=1):
        self.__weights = weights
        self.__learnRate = learnRate

    def getWeights(self):
        return self.__weights

    def train(self, xArr, y):
        approxY = np.dot(self.__weights, xArr)
        learnResult = self.__learnRate * (y - approxY)
        self.__weights += np.multiply(learnResult, xArr)

    def print(self):
      print(f'weights: {self.__weights}')


# %%
samplesTime = getFileData("data/Ex1_t")
samplesInput = getFileData("data/Ex1_x")
samplesOutput = getFileData("data/Ex1_y")
nSamples = len(samplesInput)
nTrainSamples = round(0.7*nSamples)
nTestSamples = nSamples-nTrainSamples

# %%
adaline = Adaline()

for index in range(nTestSamples):
  adaline.train([samplesInput[index]], samplesOutput[index])

adaline.print()


# %%
