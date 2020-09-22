# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from RBF import RBFN
from sklearn.model_selection import train_test_split
from generate_data import generateExerciseData

#%%
def getSeparationSurfaceTrace(x0Limits, x1Limits, evalFunction):
    x0 = np.arange(x0Limits[0], x0Limits[1], 0.01)
    x1 = np.arange(x1Limits[0], x1Limits[1], 0.01)
    z = np.array(
        [evalFunction(np.c_[x0, x * np.ones(len(x0)), np.ones(len(x0))]) for x in x1])
    return go.Surface(x=x0, y=x1, z=z)


def getSeparationLineTrace(x0Limits, evalFunction):
    x0 = np.arange(x0Limits[0], x0Limits[1], 0.01)
    x1 = np.vectorize(evalFunction)(x0)
    return go.Scatter(x=x0, y=x1)


def runSimulation(X, y, plot=False):
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.1)
    rbf = RBFN(hidden_shape=4, sigma=1.0, outputFunction=np.sign)
    rbf.fit(XTrain, yTrain)
    yTestPoints = yTest.shape[0]
    yApprox =rbf.predict(XTest)
    nWrongPredictions = (((yTest-yApprox)**2)/4).sum()
    accuracy = (yTestPoints-nWrongPredictions)/yTestPoints

    if(plot):
        fig1 = make_subplots()
        fig1.add_trace(getSeparationSurfaceTrace([-3, 3], [-3, 3], rbf.predict))
        fig1.add_trace(go.Scatter3d(
            x=data.loc[data.group.eq(1), "x0"],
            y=data.loc[data.group.eq(1), "x1"],
            z=data.loc[data.group.eq(1), "y0"],
            mode="markers",
            name="Grupo 1"
        ))
        fig1.add_trace(go.Scatter3d(
            x=data.loc[data.group.eq(2), "x0"],
            y=data.loc[data.group.eq(2), "x1"],
            z=data.loc[data.group.eq(2), "y0"],
            mode="markers",
            name="Grupo 2"
        ))
        fig1.update_layout(
            scene=dict(
                xaxis_title='x0',
                yaxis_title='x1',
                zaxis_title='y'
            )
        )
        fig1.show()

    return accuracy


accuracies = []
nPointsPerGroup = 200
data = generateExerciseData(nPointsPerGroup)
nPoints = data["x0"].shape[0]

X = np.c_[data.loc[:, ["x0", "x1"]].to_numpy(), np.ones(nPoints)]
y = data["y0"].to_numpy()

for _ in range(10):
    accuracies.append(runSimulation(X, y, True))

accuracies = np.array(accuracies)
print(accuracies.mean())
print(accuracies.std())
