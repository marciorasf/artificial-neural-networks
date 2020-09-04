import re
import numpy as np

def getFileData(path, cols):
    data = np.loadtxt(path, usecols=cols)
    return np.array(data)

def addConstantTerm(arr):
    oneVector = np.ones((arr.shape[0], 1))
    newArr = np.concatenate((arr, oneVector), axis=1)
    return newArr