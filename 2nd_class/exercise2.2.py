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
tol = 1e-5
maxIterations = 100
trainSamplesRatio = 0.9
nSamplesPerGroup = 200

# %% Initialize data
firstGroupSigma = 0.4
firstGroupMean = [2, 2]
firstGroupInput = np.random.multivariate_normal(
    firstGroupMean, [[firstGroupSigma, 0], [0, firstGroupSigma]], nSamplesPerGroup
)
firstGroupInput = utils.addConstantTerm(firstGroupInput)
firstGroupOutput = [1] * nSamplesPerGroup

secondGroupSigma = 0.4
secondGroupMean = [4, 4]
secondGroupInput = np.random.multivariate_normal(
    secondGroupMean, [[secondGroupSigma, 0], [0, secondGroupSigma]], nSamplesPerGroup
)
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

# fig1 = px.scatter(x=inputData[:, 0], y=inputData[:, 1], color=adalineApproxYArr)

# fig1.show()

# fig2 = px.scatter(x=inputData[:, 0], y=inputData[:, 1], color=outputData)

# fig2.show()

# fig3 = px.line(x=xPlan, y=yPlan)

# fig3.show()

weights = adaline.getWeights()


def hyperPlan(x):
    return -(weights[0] * x + weights[2]) / weights[1]


xPlan = np.linspace(0, 6, 100)
yPlan = np.vectorize(hyperPlan)(xPlan)

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(
    x=inputData[:, 0], 
    y=inputData[:, 1], 
    mode="markers", 
    marker=dict(color=adalineApproxYArr, colorscale='Viridis')
    ))
fig.add_trace(go.Scatter(x=xPlan, y=yPlan, mode="lines",))

fig.show()


# %%
adaline.printDetails()

# %%
