import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from knn import KNN
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


list = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass"]
output = ["R1", "2", "3", "4", "5", "6", "7"]
output2 = ["P1", "2", "3", "4", "5", "6", "7"]
matrix = pd.DataFrame(np.zeros(49).reshape(7,7),columns =output, index=output2)

file = pd.read_csv('glass.data', names = list)
file = file.sample(frac=1)
acc=[]
acclib=[]

Y = file['Glass'].values
del file['Glass']
del file['ID']
X = file.iloc[:-1].values

# print(X)
# print(Y)

train = round(214*0.6)
test = round(214*0.2)
X_train = X[:train]
X_test = X[train:train + test]
Y_train = Y[:train]
Y_test = Y[train:train + test]

# for i in range(1,len(X_train)):
#     clf = KNN(i)
#     clf.fit(X_train, Y_train)
#     predictions = clf.predict(X_test)
#     acc.append(np.sum(predictions == Y_test) / len(Y_test))

i = 4
clf = KNN(i)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
acc.append(np.sum(predictions == Y_test) / len(Y_test))
print(acc)

print(predictions)
print(Y_test)
print(matrix)

for i in range(len(Y_test)):
    matrix.iloc[[Y_test[i]-1],[predictions[i-1]-1]] += 1

print(matrix)

# for i in range(1,len(X_train)):
#     clf = KNeighborsClassifier(n_neighbors=i)
#     clf.fit(X_train, Y_train)
#     predictionslib = clf.predict(X_test)
#     acclib.append(np.sum(predictionslib == Y_test) / len(Y_test))

# plt.figure(1)
# plt.xlabel("K num")
# plt.ylabel("accuracy")
# plt.scatter(range(1,len(X_train)), acc)
# plt.show()

# plt.figure(2)
# plt.xlabel("K num")
# plt.ylabel("accuracy")
# plt.scatter(range(1,len(X_train)), acclib)
# plt.show()
