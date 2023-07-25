import numpy as np


def step(x):
    return np.where(x > 0 , 1, 0)

class Perceptron:

    def __init__(self, learning_rate=0.001, n_iters=500):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.klasy = []
        self.bias = 0

    def fit(self, X, y, klasa):
        self.klasy.clear()
        acc = 0
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for i in y:
            if i == klasa: self.klasy.append(1)
            else: self.klasy.append(0)

        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                scalar = np.dot(x_i, self.weights)
                linear_output = scalar + self.bias
                y_predicted = step(linear_output)
                update = self.lr * (self.klasy[idx] - y_predicted) 
                self.weights += update * x_i
                self.bias += update
        

    def predict(self, X):
        scalar = np.dot(X, self.weights)
        output = scalar + self.bias
        return output