# %%
import numpy as np
import pandas as pd


def generateData(mean, sigma, size, transformFunc):
    xData = np.random.normal(mean, sigma, (size, 2))
    xDim = xData.shape[1]
    xColumns = [f"x{i}" for i in range(xDim)]

    yData = np.atleast_2d(np.array(list(map(transformFunc, xData))))
    if yData.shape[0] == 1:
        yData = yData.T

    yDim = yData.shape[1]
    yColumns = [f"y{i}" for i in range(yDim)]

    data = np.c_[xData, yData]
    columns = xColumns + yColumns

    dfData = pd.DataFrame(data, columns=columns)
    return dfData


def generateExerciseData(pointsPerGroup):
    firstGroupMean = [[-1, -1], [1, 1]]
    firstGroupSigma = 0.4

    secondGroupMean = [[-1, 1], [1, -1]]
    secondGroupSigma = 0.4

    df1 = generateData(firstGroupMean[0], firstGroupSigma,
                       pointsPerGroup, lambda x: 1)
    df2 = generateData(firstGroupMean[1], firstGroupSigma,
                       pointsPerGroup, lambda x: 1)
    df3 = generateData(secondGroupMean[0], secondGroupSigma,
                       pointsPerGroup, lambda x: -1)
    df4 = generateData(secondGroupMean[1], secondGroupSigma,
                       pointsPerGroup, lambda x: -1)

    df1["group"] = 1
    df2["group"] = 1
    df3["group"] = 2
    df4["group"] = 2

    df = pd.concat([df1, df2, df3, df4])

    return df
