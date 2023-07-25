import numpy as np


def step(x):
    return np.where(x > 0 , 1, 0)

class Perceptron1:

    def __init__(self, learning_rate=0.1, n_iters=500):
        self.lr = learning_rate
        self.n_iters = n_iters

    def fit(self, X, y, i):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for klasa in y:
            if klasa == i: y[klasa.index] = 1
            else: y[klasa.index] = 0


        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                scalar = np.dot(x_i, self.weights)
                linear_output = scalar + self.bias
                y_predicted = step(linear_output)
                update = self.lr * (y[idx] - y_predicted) 
                self.weights += update * x_i
                self.bias += update


    def predict(self, X):
        scalar = np.dot(X, self.weights)
        #print(scalar)
        output = scalar + self.bias
        y_predicted = step(output)
        return y_predicted