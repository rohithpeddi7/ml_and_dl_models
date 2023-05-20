import numpy as np

class LogisticRegression():
    def __init__(self):
        pass

    def fit(self, trainX, trainY, learning_rate = 0.001, epochs = 10000):
        """
        Function to train the logistic model. Just pass in trainX, trainY and it will do the rest.
        """
        trainX = np.array(trainX)
        trainY = np.array(trainY)
        self.w = np.random.random(trainX[0].shape)
        self.b = trainY[0]
        self.learning_rate = learning_rate
        self.trainX = trainX
        self.trainY = trainY
        self.m = trainX.shape[0]
        self.history = []
        for _ in range(epochs):
            self.history.append(self._cost_function())
            self._gradient_descent()
        return self.history
    