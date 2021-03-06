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


def writeDataToCSV(df, filename):
    df.to_csv(filename)


def readDataFromCSV(filename):
    data = pd.read_csv(filename, index_col=0)
    return data


def generateExerciseData(pointsPerGroup):
    firstGroupMean = [2, 2]
    firstGroupSigma = 0.4

    secondGroupMean = [4, 4]
    secondGroupSigma = 0.4

    df1 = generateData(firstGroupMean, firstGroupSigma, pointsPerGroup, lambda x: 1)
    df2 = generateData(secondGroupMean, secondGroupSigma, pointsPerGroup, lambda x: 0)

    df1["group"] = 1
    df2["group"] = 2

    df = pd.concat([df1, df2])

    return df
