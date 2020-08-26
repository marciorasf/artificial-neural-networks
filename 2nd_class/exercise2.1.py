# %% Imports
import numpy as np
import sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import random
from adaline import Adaline
import utils

sys.path.insert(1, "../utils")
from separateIndexesByRatio import separateIndexesByRatio

# %% Initialize parameters
adalineDimension = 3
tol = 1e-15
maxIterations = 100
trainSamplesRatio = 0.9
nSamplesPerGroup = 200

# %% Initialize data
firstGroupSigma = 0.4
firstGroupMean = 2

firstGroupInput = np.random.normal(firstGroupMean, firstGroupSigma, (nSamplesPerGroup,2))
firstGroupInput = utils.addConstantTerm(firstGroupInput)

firstGroupOutput = [1] * nSamplesPerGroup

secondGroupSigma = 0.4
secondGroupMean = 4
secondGroupInput = np.random.normal(secondGroupMean, secondGroupSigma, (nSamplesPerGroup,2))
secondGroupInput = utils.addConstantTerm(secondGroupInput)

secondGroupOutput = [-1] * nSamplesPerGroup

inputData = np.concatenate((firstGroupInput, secondGroupInput))
outputData = np.concatenate((firstGroupOutput, secondGroupOutput))

trainIndexes, testIndexes = separateIndexesByRatio(
    2 * nSamplesPerGroup, trainSamplesRatio
)
random.shuffle(trainIndexes)

# %% Initialize and Train Adaline

adaline = Adaline([0] * (adalineDimension), 0.1, lambda x: 1 if x >= 0 else -1)

xTrain = inputData[trainIndexes]
yTrain = outputData[trainIndexes]
adaline.train(xTrain, yTrain, tol, maxIterations)


# %% Test
xTest = inputData[testIndexes]
yTest = outputData[testIndexes]
testResult = adaline.test(xTest, yTest)
print(f"Mean Squared Error: {testResult}")


# %% Plot
adalineApproxYArr = adaline.evaluate(inputData)

weights = adaline.getWeights()

def hyperPlan(x):
    return -(weights[0] * x + weights[2]) / weights[1]

xPlan = np.linspace(0.5, 5.5, 100)
yPlan = np.vectorize(hyperPlan)(xPlan)

fig = make_subplots(x_title="x", y_title="y")

# Add traces
fig.add_trace(go.Scatter(
    x=firstGroupInput[:, 0], 
    y=firstGroupInput[:, 1], 
    mode="markers", 
    name="Grupo 1"
    ))
fig.add_trace(go.Scatter(
    x=secondGroupInput[:, 0], 
    y=secondGroupInput[:, 1], 
    mode="markers", 
    name="Grupo 2"
    ))
fig.add_trace(go.Scatter(x=xPlan, y=yPlan, mode="lines",name="Hiperplano separador"))

fig.show()

# %%
adaline.printDetails()
