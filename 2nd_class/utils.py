import re
import numpy as np

def getFileData(path, cols):
    data = np.loadtxt(path, usecols=cols)
    return np.array(data)


def getFileDataAlt(path):
    data = open(path, "r").read()
    data = re.sub(r'"\w*"\n?|| ', "", data)
    data = data.split("\n")
    data.pop()
    data = [float(val) for val in data]
    return data


def addConstantTerm(arr):
    oneVector = np.ones((arr.shape[0], 1))
    newArr = np.concatenate((arr, oneVector), axis=1)
    return newArr