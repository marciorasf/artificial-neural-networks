#  %%
import numpy as np
import pandas as pd
from generate_data import generateExerciseData
from mcp import Mcp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# %%
data = generateExerciseData(200)
perc = Mcp(2, "perceptron")

xMatrix = data.loc[:, ["x0", "x1"]].to_numpy()
yVector = data.loc[:, "y0"].to_numpy()

perc.train(xMatrix, yVector, 1e-6, 1000)
testResult = perc.test(xMatrix, yVector)
print(f"test result = {testResult}")
perc.printDetails()

yApprox = perc.evaluate(xMatrix)


# %%
def getSeparationSurfaceTrace(x0Limits, x1Limits, evalFunction):
    x0 = np.arange(x0Limits[0], x0Limits[1], 0.01)
    x1 = np.arange(x1Limits[0], x1Limits[1], 0.01)
    z = np.array([evalFunction(np.c_[x0, x * np.ones(len(x0))]) for x in x1])
    return go.Surface(x=x0, y=x1, z=z)


fig = make_subplots()
fig.add_trace(getSeparationSurfaceTrace([0, 6], [0, 6], perc.evaluate))
fig.add_trace(
    go.Scatter3d(
        x=xMatrix[:, 0], y=xMatrix[:, 1], z=yVector, mode="markers"
    )
)
fig.show()

fig = px.scatter(data, x="x0", y="x1", color="group")
fig.show()

