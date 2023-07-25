import numpy as np
from collections import Counter

def distance(x1, x2):
    dis = np.sqrt(np.sum((x1-x2)**2))
    return dis

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
      
    def predict (self, X):
        predictions = [self.predict_one(x) for x in X]
        return predictions
    
    def predict_one(self, x):

        distances = [distance(x, x_train) for x_train in self.X_train]
        k_ind = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_ind]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    
        