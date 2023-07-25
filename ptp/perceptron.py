import numpy as np

def unit_step_func(x):
    return np.sign(x)
    

class ptp:

    def __init__(self, learning_rate=0.01, n_iters=500):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iters):
            for idx, x_j in enumerate(X):
                 output = np.dot(x_j, self.w) + self.bias
                 predicted = unit_step_func(output)
                 update = self.lr * (y[idx] - predicted)
                 self.weights += update * x_j
                 self.bias += update

