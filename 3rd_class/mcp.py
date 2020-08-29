# %%
import numpy as np


def addOnesColumn(matrix):
    dim = matrix.shape[0]
    return np.c_[np.ones(dim), matrix]


def getActivationFunction(name):
    if name == "perceptron":
        return lambda x: 1 if x >= 0 else 0
    else:
        return lambda x: x


class Mcp:
    def __init__(
        self,
        xDimension,
        activationFunction="adaline",
        learningRate=0.01,
        initialWeights=None,
    ):
        self.xDimension = xDimension
        if type(activationFunction) == "function":
            self.activationFunction = activationFunction
        else:
            self.activationFunction = getActivationFunction(activationFunction)

        self.learningRate = learningRate

        if not initialWeights:
            initialWeights = np.zeros(xDimension + 1)

        self.weights = initialWeights
        self.nTrainings = 0

    def evaluate(self, xMatrix):
        xMatrix = np.atleast_2d(xMatrix)
        xMatrix = addOnesColumn(xMatrix)
        return self.__internalEvaluate(xMatrix)

    def __internalEvaluate(self, xMatrix):
        linearCombination = np.dot(xMatrix, self.weights)
        yVector = np.vectorize(self.activationFunction)(linearCombination)
        return yVector

    def train(self, xMatrix, yVector, tolerance=1e-3, maxIterations=1):
        xMatrix = addOnesColumn(xMatrix)
        for _ in range(maxIterations):
            iterationError = 0
            for index, xRow, in enumerate(xMatrix):
                yApprox = self.__internalEvaluate(xRow)
                error = yVector[index] - yApprox
                self.weights = self.weights + self.learningRate * error * xRow
                iterationError += error ** 2
                self.nTrainings += 1

            iterationError /= xMatrix.shape[0]
            if iterationError <= tolerance:
                break

    def test(self, xMatrix, yVector):
        yApprox = self.evaluate(xMatrix)
        return self.calcError(yVector, yApprox)

    def calcError(self, y, yApprox):
        return np.linalg.norm((y - yApprox), ord=2)

    def getWeights(self):
        return self.weights

    def printDetails(self):
        print(f"weights = {self.weights}\ntrained {self.nTrainings} times")
