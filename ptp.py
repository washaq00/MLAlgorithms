import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from perceptron1 import Perceptron
from collections import Counter


list = ["1", "2",'3','4','ID']
output = ["R1", "2"]
output2 = ["P1", "2"]
matrix = pd.DataFrame(np.zeros(4).reshape(2,2),columns =output, index=output2)

file = pd.read_csv('bk.txt', names = list)
file = file.sample(frac=1)
acc=[]

Y = file['ID'].values
del file['ID']
X = file.iloc[:].values

train = round(len(Y) *0.6)
test = round(len(Y)*0.2)
X_train = X[:train]
X_test = X[train:train + test]
Y_train = Y[:train]
Y_test = Y[train:train + test]

clf = Perceptron()
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
acc = (np.sum(predictions == Y_test) / len(Y_test))
print(acc)
print(matrix)

print(len(Y_test))
for i in range(len(Y_test)):
    matrix.iloc[[Y_test[i]],[predictions[i]]] += 1

print(matrix)
