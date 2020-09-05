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
    z = np.array([evalFunction(np.c_[x0, x * np.ones(len(x0))]) for x in x1])
    return go.Surface(x=x0, y=x1, z=z)


def getSeparationLineTrace(x0Limits, evalFunction):
    x0 = np.arange(x0Limits[0], x0Limits[1], 0.01)
    x1 = np.vectorize(evalFunction)(x0)
    return go.Scatter(x=x0, y=x1)


# %% generate input data
nNeurons = 20
xDim = 2
nPoints = 200

data = generateExercise1Data(nPoints)
X = data.loc[:, ["x0", "x1"]].to_numpy()
X = np.c_[X, np.ones(2 * nPoints)]
Y = data["y0"].to_numpy()


# %%
Z = np.random.uniform(size=(nNeurons, xDim+1))

H = np.matmul(X, Z.T)
HPseudoInverse = np.linalg.pinv(H)

W = np.matmul(HPseudoInverse, Y)

# %%

YApprox = np.array(
    list(map(lambda x: 1 if x == 1 else -1, np.sign(np.matmul(H, W))))
)

testResult = (((Y-YApprox)**2)/4).sum()
print(testResult)
# %%
fig1 = px.scatter_3d(data, x="x0", y="x1", z="y0", color="group")
fig1.show()

fig2 = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=YApprox, color=data["group"])
fig2.show()
