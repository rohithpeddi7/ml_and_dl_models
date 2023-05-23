import numpy as np

class LogisticRegression():
    def __init__(self):
        pass

    def fit(self, trainX, trainY, learning_rate = 0.0001, epochs = 10000):
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
    
    def _cost_function(self):
        """
        Categorical cross entropy cost function
        """
        pred_y = self.predict(self.trainX)
        return np.sum((-self.trainY*np.log(pred_y+1e-9) -(1-self.trainY)*np.log(1+1e-9-pred_y)))/(self.m)
    
    def predict(self, test_X):
        """
        Predicts the output for the given input
        """
        test_x = np.array(test_X)
        test_x = test_x.reshape((-1,self.w.shape[0]))
        return 1/(1+np.exp(-(np.dot(self.w,test_x.T)+self.b)))

    def _gradient_descent(self):
        """
        Does gradient descent with the given learning rate
        """
        pred_y = self.predict(self.trainX)
        self.w = self.w - self.learning_rate*((pred_y-self.trainY).reshape((-1,1))*self.trainX).sum(axis=0)/self.m
        self.b = self.b - self.learning_rate*np.sum(pred_y - self.trainY)/self.m

        
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer as lbc
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = lbc()
    trainX, testX, trainY, testY = train_test_split(data.data, data.target, test_size=0.2)
    meow = LogisticRegression()
    meow.fit(trainX, trainY, learning_rate=0.0001, epochs=10000)
    print(accuracy_score(testY, meow.predict(testX)>0.5))
    
    
    