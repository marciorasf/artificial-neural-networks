# %%
import numpy as np
import pandas as pd


def generateData(mean, cov, size, transformFunc):
    xData = np.atleast_2d(np.random.multivariate_normal(mean, cov, size))
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


def writeDataToCSV(df, filename):
    df.to_csv(filename)


def readDataFromCSV(filename):
    data = pd.read_csv(filename, index_col=0)
    return data


def generateExerciseData():
    pointsPerGroup = 200

    firstGroupMean = [2, 2]
    firstGroupCov = [[0.4, 0], [0, 0.4]]

    secondGroupMean = [4, 4]
    secondGroupCov = [[0.4, 0], [0, 0.4]]

    df1 = generateData(firstGroupMean, firstGroupCov, pointsPerGroup, lambda x: 1)
    df2 = generateData(secondGroupMean, secondGroupCov, pointsPerGroup, lambda x: -1)

    df1["group"] = 1
    df2["group"] = 1

    df = pd.concat([df1, df2])

    return df
