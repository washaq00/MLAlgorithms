from perceptron import Perceptron
import numpy as np
import pandas as pd

class OneVsRest():
    def __init__(self, klasa):
        self.perceptrony = [Perceptron() for i in klasa]
        self.klasy = []

    def wczytajDane(self, X, Y):
        self.klasy.clear()
        for klasa in Y:
            if klasa != '': self.klasy.append(klasa)
       
        for per in self.perceptrony:
            per.fit(X, Y, self.perceptrony.index(per)+1)

    def wywolaj(self, X):
        predictions = [[] for i in range(7)]
        for per in range(len(self.perceptrony)):
            predictions[per]= self.perceptrony[per].predict(X)
        
        temp=[]
        wynikifinalne = []
        for y in range(len(predictions[0])):
            temp.clear()
            temp.append(predictions[0][y])
            temp.append(predictions[1][y])
            temp.append(predictions[2][y])
            temp.append(predictions[4][y])
            temp.append(predictions[5][y])
            temp.append(predictions[6][y])
            index = np.argmax(temp) + 1
            wynikifinalne.append(index)
        return wynikifinalne
    
    def analiza(self, X,Y):
         output = ["R1", "2", "3", "4", "5", "6", "7"]
         output2 = ["P1", "2", "3", "4", "5", "6", "7"]
         matrix = pd.DataFrame(np.zeros(49).reshape(7,7),columns =output, index=output2)
         print(Y)
         wyniki = self.wywolaj(X)
         print(wyniki)
         acc = (np.sum(wyniki == Y) / len(Y))
         for i in range(len(Y)):
            matrix.iloc[[Y[i]-1],[wyniki[i]-1]] += 1
         print("OVR Macierz i dokladnosc")
         print(acc)
         print(matrix)
         