# %%
import numpy as np
import pandas as pd
from RBF import RBFN
from sklearn import datasets
import plotly.express as px
import plotly.graph_objects as go

# %% declares functions


def incrementIndexesInterval(indexes, step):
    indexes = indexes + np.array([step, step])
    indexes = [round(index) for index in indexes]
    return indexes


def crossValidation(model, data, target, folds):
    nRows = len(target)
    percentual_error = []
    step = nRows / folds
    testIndexesInterval = np.array([0, round(step)-1])

    for _ in range(folds):
        model.reset()
        testIndexes = np.arange(
            testIndexesInterval[0], testIndexesInterval[1],  dtype=np.int32)
        trainIndexes = np.array(
            [i for i in range(nRows) if i not in testIndexes], dtype=np.int32)

        trainData = data[trainIndexes]
        trainTarget = target[trainIndexes]
        testData = data[testIndexes]
        testTarget = target[testIndexes]

        model.fit(trainData, trainTarget)
        yApprox = model.predict(testData)

        testIndexesInterval = incrementIndexesInterval(
            testIndexesInterval, step)

        percentual_error.append(sum(((yApprox - testTarget)**2))/len(testData))

    return percentual_error

#  %% initialize data
rawData = datasets.load_breast_cancer()

dataColumns = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
]
targetColumn = "target"

df = pd.DataFrame(rawData["data"][:, 0:9], columns=dataColumns)
df[targetColumn] = rawData["target"]
df = df.sample(frac=1).reset_index(drop=True)


# %%
def runSimulation(hidden_shape):
    rbf = RBFN(hidden_shape=hidden_shape, sigma=1.0, outputFunction=np.sign)

    error = np.array(crossValidation(
        rbf,
        np.array(df[dataColumns]),
        np.array(df[targetColumn]),
        10,
    ))

    return {"mean": error.mean(), "std":error.std()}

errors = {}
for i in range(5, 105, 5):
    errors[i] = runSimulation(i)

# %%

means =[]
stds =[]
for value in errors.values():
    print(value)
    means.append(value["mean"]*100)
    stds.append(value["std"]*100)

fig = go.Figure()
fig.add_trace(go.Bar(
    x= list(errors.keys()),
    y=means,
    name='Erro Percentual Médio',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x= list(errors.keys()),
    y=stds,
    name='Desvio Padrão Médio',
    marker_color='lightsalmon'
))
fig.update_yaxes(title_text="%")
fig.update_xaxes(title_text="Número de funções")
fig.show()
