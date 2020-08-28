# %%
import numpy as np


def addOnesColumn(matrix):
    dim = matrix.shape[1]
    return np.c_[np.ones(dim), matrix]


class Perceptron:
    def __init__(
        self,
        xDimension,
        activationFunction=lambda x: 1 if x >= 0 else 0,
        initialWeights=None,
    ):
        self.xDimension = xDimension
        self.activationFunction = activationFunction

        if not initialWeights:
            initialWeights = np.zeros(xDimension + 1)

        self.weights = initialWeights
        self.nTrainings = 0

    def evaluate(self, xMatrix):
        xMatrix = addOnesColumn(xMatrix)
        linearCombination = np.dot(xMatrix, self.weights)
        yVector = np.vectorize(self.activationFunction)(linearCombination)
        return yVector

    def train(self, xMatrix, yVector):
        pass

    def test(self, xMatrix, yVector):
        pass


perc = Perceptron(2, lambda x: x + 1)
x = np.array([[1, 2], [1, 3]])
y = perc.evaluate(x)

