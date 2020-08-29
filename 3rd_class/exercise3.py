# %%
import numpy as np
import pandas as pd
from mcp import Mcp
from sklearn import datasets

# %% declares functions


def incrementIndexesInterval(indexes, step):
  indexes = indexes + np.array([step, step])
  indexes = [round(index) for index in indexes]
  return indexes


def crossValidation(model, data, target, folds, tolerance=1e-3, maxIterations=1):
  nRows = len(target)
  scores = []
  weights = []
  step = nRows / folds
  testIndexesInterval = np.array([0, round(step)-1])

  for _ in range(folds):
    model.resetWeights()
    testIndexes = np.arange(
      testIndexesInterval[0], testIndexesInterval[1],  dtype=np.int32)
    trainIndexes = np.array(
      [i for i in range(nRows) if i not in testIndexes], dtype=np.int32)

    trainData = data[trainIndexes]
    trainTarget = target[trainIndexes]
    testData = data[testIndexes]
    testTarget = target[testIndexes]

    model.train(trainData, trainTarget, tolerance, maxIterations)
    scores.append({"accuracy":model.test(testData, testTarget), "nData": len(testTarget) })
    weights.append(model.getWeights())

    testIndexesInterval = incrementIndexesInterval(testIndexesInterval, step)

  return [scores, weights]


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
perc = Mcp(9, "perceptron", "classification")

scores, weights = crossValidation(perc, np.array(
  df[dataColumns]), np.array(df[targetColumn]), 10, 1e-3, 100)

print(scores)
print([list(weight) for weight in weights])
# %%
