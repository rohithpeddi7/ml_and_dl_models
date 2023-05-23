import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class LinearRegression:
    def __init__(self):
        pass

    def fit(self, trainX, trainY, learning_rate = 0.001, epochs = 10000):
        """
        Function to train the linear model. Just pass in trainX, trainY and it will do the rest.
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
        for _ in tqdm(range(epochs)):
            self.history.append(self._cost_function())
            self._gradient_descent()
        return self.history

    def _cost_function(self):
        """
        Root mean squared error loss function
        """
        return np.sum(np.power(np.dot(self.w,self.trainX.T)+self.b-self.trainY,2))/(2*self.m)

    def _gradient_descent(self):
        """
        Does gradient descent with the given learning rate
        w_i = w_i - alpha*dJ/dw_i  ==> w_i = w_i - alpha*1/m*sigma(w.x+b-y)*x
        """
        c = np.sum(np.multiply((np.dot(self.w,self.trainX.T)+self.b-self.trainY).reshape((-1,1)), self.trainX),axis=0)/(self.m)
        self.w = self.w - self.learning_rate*c
        self.b = self.b - self.learning_rate*np.sum((np.dot(self.w,self.trainX.T)+self.b-self.trainY))/self.m

    def predict(self, test_X):
        test_x = np.array(test_X)
        test_x = test_x.reshape((-1,self.w.shape[0]))
        return np.dot(self.w,test_x.T)+self.b
        
if __name__ == '__main__':
    # ok = LinearRegression()
    # _train_x = [[1,2], [3,10], [7,6], [20,3], [9,8], [-2,-4], [1,100]]
    # _train_y = [7, -3, 17, 62, 19, 10, -189]
    # ok.fit(_train_x,_train_y)
    # print(ok.w,ok.b)
    # print(ok.predict([10,2]))
    from sklearn.datasets import fetch_california_housing as load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler

    data = load_boston()
    scaler = StandardScaler()
    data.data = scaler.fit_transform(data.data)
    trainX, testX, trainY, testY = train_test_split(data.data, data.target, test_size=0.2)
    meow = LinearRegression()
    meow.fit(trainX, trainY, learning_rate=0.001, epochs=6000)
    print(mean_squared_error(testY, meow.predict(testX)))

