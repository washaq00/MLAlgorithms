import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from knn import KNN
from collections import Counter
from OVR import OneVsRest
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB


list = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass"]
klasy = [i for i in range(1,8)]

def macierz(predykcje, target):
    output = ["R1", "2", "3", "4", "5", "6", "7"]
    output2 = ["P1", "2", "3", "4", "5", "6", "7"]
    matrix = pd.DataFrame(np.zeros(49).reshape(7,7),columns =output, index=output2)
    acc = np.sum(predykcje == target) / len(target)
    print(acc)
    for i in range(len(target)):
      matrix.iloc[[target[i]-1],[predykcje[i-1]-1]] += 1
    print(matrix)

file = pd.read_csv('Xtest.data', names = list)
file2 = pd.read_csv('Xtrain.data', names = list)

Y_test = file['Glass'].values
file['Mg'] = file["Mg"]/3
file['Na'] = file["Na"]/10
file['Si'] = file["Si"]/70
file['Ca'] = file["Ca"]/9
del file['Glass']
del file['ID']
X_test = file.values

Y_train = file2['Glass'].values
file2['Mg'] = file2["Mg"]/3
file2['Na'] = file2["Na"]/10
file2['Si'] = file2["Si"]/70
file2['Ca'] = file2["Ca"]/9
del file2['Glass']
del file2['ID']
X_train = file2.values

clf = OneVsRest(klasy)
clf.wczytajDane(X_train, Y_train)
clf.analiza(X_test,Y_test)

acclib3=[]

print("\nMLP")

mlp = MLPClassifier(max_iter=5000, hidden_layer_sizes=(100,100))
mlp.fit(X_train,Y_train)
predict_test = mlp.predict(X_test)
macierz(predict_test, Y_test)

# for i in range(100,500):
#   mlp = MLPClassifier(max_iter=i, hidden_layer_sizes=(100,100))
#   mlp.fit(X_train,Y_train)
#   predict_test = mlp.predict(X_test)
#   acclib3.append(np.sum(predict_test == Y_test) / len(Y_test))

# plt.figure(1)
# plt.xlabel("max iter")
# plt.ylabel("accuracy")
# plt.scatter(range(100,500), acclib3)
# plt.show()

# macierz(predict_test,Y_test)

print("\nDecision Tree Macierz i dokladnosc")
acclib2=[]
clf3 = tree.DecisionTreeClassifier(max_depth=4, max_leaf_nodes= 10)
clf3.fit(X_train, Y_train)
predict2 = clf3.predict(X_test)

for i in range(2,100):
     clf3 = tree.DecisionTreeClassifier(max_leaf_nodes=i)
     clf3.fit(X_train, Y_train)
     predict2 = clf3.predict(X_test)
     acclib2.append(np.sum(predict2 == Y_test) / len(Y_test))

plt.figure(2)
plt.xlabel("max leaf")
plt.ylabel("accuracy")
plt.scatter(range(2,100), acclib2)
plt.show()


# macierz(predict2,Y_test)

# print("\nNaive Bayes Macierz i dokladnosc")

# model = MultinomialNB(alpha=0)
# model.fit(X_train, Y_train)
# predicted = model.predict(X_test)
# print(predicted)

# macierz(predicted,Y_test)

# acclib=[]

# for i in range(1,len(X_test)):
#      clf = KNN(i)
#      clf.fit(X_train, Y_train)
#      predictions = clf.predict(X_test)
#      acclib.append(np.sum(predictions == Y_test) / len(Y_test))

# plt.figure(3)
# plt.xlabel("K num")
# plt.ylabel("accuracy")
# plt.scatter(range(1,len(X_test)), acclib)
# plt.show()
