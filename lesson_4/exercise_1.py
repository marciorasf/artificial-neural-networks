# %% imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


from generate_data import generateExercise1Data, generateExercise2Data

# %% Declare functions


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


def elm(nNeurons, data, plot):
    nPoints = data.loc[:, "x0"].shape[0]
    xDim = 2

    X = np.c_[data.loc[:, ["x0", "x1"]].to_numpy(), np.ones(nPoints)]
    Y = data["y0"].to_numpy()

    Z = np.random.uniform(size=(nNeurons, xDim+1))

    H = np.matmul(X, Z.T)
    HPseudoInverse = np.linalg.pinv(H)

    W = np.matmul(HPseudoInverse, Y)

    def evalInput(X):
        result = np.sign(np.matmul(np.matmul(X, Z.T), W))
        return list(map(lambda x: 1 if x >= 0 else -1, result))

    YApprox = evalInput(X)

    nWrongYAprox = (((Y-YApprox)**2)/4).sum()
    accuracy = (nPoints-nWrongYAprox)/nPoints

    if(plot):
        fig1 = make_subplots()
        fig1.add_trace(getSeparationSurfaceTrace([0, 6], [0, 6], evalInput))
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

acc = []
for _ in range(100):
    data = generateExercise1Data(200)
    acc.append(elm(6, data, False))

acc = np.array(acc)

print(acc.mean())
print(acc.std())
