# %%
import numpy as np
import plotly.express as px
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


def getRandomArray(xStart, xEnd, nSamples):
    return [np.random.uniform(xStart, xEnd) for _ in range(nSamples)]


def addGaussianNoise(arr, mu, sigma):
    return list(map(lambda value: value + gaussianNoise(mu, sigma), arr))


def gaussianNoise(mu, sigma):
    return np.random.normal(mu, sigma, 1)[0]


def evaluateFunction(func, iterable):
    return list(map(func, iterable))


def getH(xArr, polyDegree):
    def calculateHLine(x):
        return list(map(lambda degree: pow(x, degree), range(polyDegree+1)))

    return list(map(calculateHLine, xArr))


def getWeights(xArr, yArr, polyDegree):
    H = getH(xArr, polyDegree)
    HPseudoInverse = np.linalg.pinv(H)
    return np.matmul(HPseudoInverse, yArrNoisy)


def getApproxPoly(xArr, yArr, polyDegree):
    weights = getWeights(xArr, yArrNoisy, polyDegree)
    return np.polynomial.Polynomial(weights)


# %%
xStart, xEnd = -15, 10
nSamples = 1000
mu, sigma = 0, 4

originalPoly = np.polynomial.Polynomial([10, 3, 0.5])


# %%
xArr = getRandomArray(xStart, xEnd, nSamples)
yArr = evaluateFunction(originalPoly, xArr)
yArrNoisy = addGaussianNoise(yArr, mu, sigma)

px.scatter(x=xArr, y=yArrNoisy)

# %%


def subRoutine(polyDegree):
    approxPoly = getApproxPoly(xArr, yArrNoisy, polyDegree)
    yApprox = evaluateFunction(approxPoly, xArr)
    print("degree: " + str(polyDegree))
    print("coefs:", str(approxPoly.coef))
    px.scatter(x=xArr, y=yApprox)


# %%
for degree in range(1, 3):
    subRoutine(degree)
