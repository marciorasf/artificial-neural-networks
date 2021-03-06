#%%
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# %%
import numpy as np
import plotly.express as px


# %%
xStart, xEnd = -15, 10
nSamples = 10
mu, sigma = 0, 4

originalPoly = np.polynomial.Polynomial([10, 3, 0.5])


def getRandomArray(xStart, xEnd, nSamples):
    return np.array([np.random.uniform(xStart, xEnd) for _ in range(nSamples)])


def addGaussianNoise(arr, mu, sigma):
    return np.array(list(map(lambda value: value + gaussianNoise(mu, sigma), arr)))


def gaussianNoise(mu, sigma):
    return np.random.normal(mu, sigma, 1)[0]


def evaluateFunction(func, iterable):
    return np.array(list(map(func, iterable)))


def getH(xArr, polyDegree):
    def calculateHLine(x):
        return np.array(list(map(lambda degree: pow(x, degree), range(polyDegree + 1))))

    return np.array(list(map(calculateHLine, xArr)))


def getWeights(xArr, yArr, polyDegree):
    H = getH(xArr, polyDegree)
    HPseudoInverse = np.linalg.pinv(H)
    return np.matmul(HPseudoInverse, yArrNoisy)


def getApproxPoly(xArr, yArr, polyDegree):
    weights = getWeights(xArr, yArrNoisy, polyDegree)
    return np.polynomial.Polynomial(weights)


def plotPoly(poly, xStart, xEnd, nSamples):
    xArr = np.linspace(xStart, xEnd, nSamples)
    yArr = evaluateFunction(poly, xArr)
    fig = px.line(x=xArr, y=yArr)
    fig.show()


# %%
def subRoutine(xArr, polyDegree):
    approxPoly = getApproxPoly(xArr, yArrNoisy, polyDegree)
    print("degree: " + str(polyDegree))
    print("coefs:", str(approxPoly.coef))
    plotPoly(approxPoly, xStart, xEnd, 1000)


# %%
plotPoly(originalPoly, xStart, xEnd, 1000)

xArr = getRandomArray(xStart, xEnd, nSamples)
yArr = evaluateFunction(originalPoly, xArr)
yArrNoisy = addGaussianNoise(yArr, mu, sigma)

px.scatter(x=xArr, y=yArrNoisy)

# %%
for degree in range(2, 3):
    subRoutine(xArr, degree)
