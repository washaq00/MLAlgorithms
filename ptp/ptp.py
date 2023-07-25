import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from perceptron import ptp
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


list = ["1", "2",'3','4','ID']

file = pd.read_csv('bk.data', names = list)
file = file.sample(frac=1)
acc=[]

print(file)

Y = file['ID'].values
del file['ID']
X = file.iloc[:].values

print(Y)
print(X)

train = round(len(Y) *0.6)
test = round(len(Y)*0.2)
X_train = X[:train]
X_test = X[train:train + test]
Y_train = Y[:train]
Y_test = Y[train:train + test]
