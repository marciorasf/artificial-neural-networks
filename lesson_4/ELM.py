import numpy as np

class ELM:
    def __init__(self, nHiddenUnits, outputFunction=lambda x:x):
        self.nHiddenUnits = nHiddenUnits
        self.outputFunction = outputFunction

    def fit(self, X, labels):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        self.randomWeights = np.random.randn(X.shape[1], self.nHiddenUnits)
        G = np.tanh(X.dot(self.randomWeights))
        self.wElm = np.linalg.pinv(G).dot(labels)

    def predict(self, X):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        G = np.tanh(X.dot(self.randomWeights))
        return np.vectorize(self.outputFunction)(G.dot(self.wElm))
