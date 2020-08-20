import numpy as np


class Adaline:
    def __init__(self, weights=[1], learnRate=0.1, activationFunction=lambda x: x):
        self.__weights = np.array(weights)
        self.__learnRate = learnRate
        self.__nTrains = 0
        self.__activationFunction = activationFunction

    def evaluate(self, xMatrix):
        return np.vectorize(self.__activationFunction)(
            np.dot(xMatrix, np.transpose(self.__weights))
        )

    def train(self, xMatrix, yArr, tol=1e-5, maxIterations=1):
        for _ in range(maxIterations):
            iterationError = 0
            for index in range(len(yArr)):
                error = self.singleTrain(xMatrix[index], yArr[index])
                iterationError += error ** 2

            if iterationError < tol:
                break

    def test(self, xMatrix, yArr):
        approxYArr = self.evaluate(xMatrix)
        meanSquaredError = ((yArr - approxYArr) ** 2).mean()
        return meanSquaredError

    def singleTrain(self, xArr, y):
        approxY = self.evaluate(xArr)
        error = y - approxY
        self.__weights = np.add(self.__weights, self.__learnRate * error * xArr)
        self.__nTrains += 1
        return error

    def printDetails(self):
        print(f"Weights: {self.__weights}\nTimes trained: {self.__nTrains}")

    def getWeights(self):
        return self.__weights
