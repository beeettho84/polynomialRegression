import numpy as np
import math

prx = np.array([[108], [115], [106], [97], [95], [91], [97], [83], [83], [78], [54], [67], [56], [53], [61], [115], [81], [78], [30], [45], [99], [32], [25], [28], [90], [89]])
#prx = [108,115,106,97,95,91,97,83,83,78,54,67,56,53,61,115,81,78,30,45,99,32,25,28,90,89]
pry = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93]
#pry = [4,5,6]
seleccion = None
indice = None
B = None

class Dataset:
    x = np.array
    x2 = np.array
    y = np.array
    matrix = list()
    def __init__(self, inx, iny):
        self.x = np.insert(inx, 0, 1, axis=1)
        unos = list()
        l1 = inx
        print(l1)
        l2 = list()
        l3 = list()
        
        for i in range(len(inx)):
            unos.append(1)
            l2.append(l1[i]**2)
            l3.append(l1[i]**3)
        #self.x2 = np.array([unos, l1, l2])

        #print(self.x2)
        #l2 = [[1,i,i**2]for i in l1]
        print("L2 = ", str(l2))
        #print("L3 = ",str(l3))
        #self.x2 = np.insert(self.x, 2, l2, axis=1) #test
        #print("X2 ",str(self.x2))
        self.y = iny
        self.matrix = toMatrix(inx, iny)

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getMatrix(self):
        return self.matrix

    def null(self):
        self.x = list()
        self.y = list()

class DiscreteMath:
    def SumX(x):
        return sum(x)

    def SumY(y):
        return sum(y)

    def SumXY(x, y):
        i=0
        out = 0.0
        if len(x) == len(y):
            for i in range(len(x)):
                out += x[i] * y[i]
            return out
        else:
            print("error, X y Y deben tener la misma longitud")

    def SumX2(x):
        i = 0
        out = 0.0
        for i in range(len(x)):
            out += x[i] ** 2
        return out

    def SumX3(x):
        i = 0
        out = 0.0
        for i in range(len(x)):
            out += x[i] ** 3
        return out

    def SumY2(y):
        i = 0
        out = 0.0
        for i in range(len(y)):
            out += y[i] * y[i]
        return out

    def SumXSumX(x):
        return sum(x)*sum(x)

    def SumXSumY(x,y):
        return sum(x)*sum(y)

class pls:
    data = None
    B0 = float()
    B1 = float()
    n = 0
    r = 0
    r2 = 0
    def __init__(self, inx, iny):
        if(len(inx) == len(iny)):
            self.n = len(inx)
            self.data = Dataset(inx, iny)
            self.calculaBs()

    def calculaBs(self):
        Sx = DiscreteMath.SumX(self.data.getX())  # obtenemos los valores parciales de las ecuaciones para facilitar el calculo
        Sy = DiscreteMath.SumY(self.data.getY())
        Sxy = DiscreteMath.SumXY(self.data.getX(), self.data.getY())
        Sx2 = DiscreteMath.SumX2(self.data.getX())
        Sy2 = DiscreteMath.SumY2(self.data.getY())
        self.B1 = (self.n * Sxy - (DiscreteMath.SumXSumY(self.data.getX(), self.data.getY()))) / (self.n * Sx2 - (DiscreteMath.SumXSumX(self.data.getX())))  # calculamos el valor de B1, necesario para obtener B0
        self.B0 = (Sy - (self.B1 * Sx)) / self.n  # calculamos B0
        Ssr = sum((yi - (self.B0 + self.B1 * xi)) ** 2 for xi, yi in zip(self.data.getX(), self.data.getY()))
        y_mean = Sy / self.n
        Sst = sum((yi - y_mean) ** 2 for yi in self.data.getY())
        self.r = ((self.n*Sxy)-(DiscreteMath.SumXSumY(self.data.getX(), self.data.getY())))/math.sqrt((self.n*Sx2-(DiscreteMath.SumXSumX(self.data.getX())))*(self.n*Sy2-(DiscreteMath.SumXSumX(self.data.getY()))))
        self.r2 = 1 - (Ssr / Sst)
        print("B0 es igual a ", self.B0)  # imprimimos los valores en consola, para implementaciones sin interfaz grafica
        print("B1 es igual a ", self.B1)

    def getB0(self): #funcion de regresion lineal simple, solicita como entrada dos arreglos
        return self.B0

    def getB1(self):
        return self.B1

    def null(self):
        self.n = 0
        self.data.null()
        self.B0 = 0.0
        self.B1 = 0.0

    def pop(self, pos):
        if pos <= self.n:
            self.data.getX().pop(pos)
            self.data.getY().pop(pos)
            self.n = len(self.data.getX())
            self.calculaBs()

    def input(self, inx, iny):
        self.data.getX().append(inx)
        self.data.getY().append(iny)
        self.n = len(self.data.getX())
        self.calculaBs()

    def predict(self, ox):
        oy = self.B0 + (ox * self.B1)
        print("Dado X = "+str(ox)+", Y = "+str(self.B0)+" + ("+str(self.B1)+" * "+str(ox)+") = "+str(oy))
        return oy

    def getR2(self):
        return self.r2

    def getR(self):
        return self.r

class prl:
    data = None
    Bs = list()
    n = 0
    r = 0
    r2 = 0

    def __init__(self, inx, iny):
        if (len(inx) == len(iny)):
            self.n = len(inx)
            self.data = Dataset(inx, iny)
            self.RegresionLineal()
    def RegresionLineal(self):
        print("X = ", str(self.data.getX()))
        Xt = np.transpose(self.data.getX())
        print("Transpuesta: ", str(Xt))
        mr = np.dot(Xt,  self.data.getX())
        print(mr)
        Xi = np.linalg.inv(mr)
        print("Xi = ", str(Xi))
        mr2 = (Xi @ Xt) @ self.data.getY()
        print("Betas: ",str(mr2))
        return mr2

    def predict(self, x):
        print("aun no hago predicciones")

    def getR2(self):

        print("aun no tengo R2")

    def getR(self):
        print("aun no tengo R")

def toMatrix(list1, list2):
    if len(list1) == len(list2):
        r = list()
        for i in range(len(list1)):
            r.append((list1[i], list2[i]))
        #print(r)
        return r


exam = prl(prx,pry)

exam.predict(24)
exam.predict(25)
exam.predict(27)
exam.predict(28)
exam.predict(29)
print("R squared = "+str(exam.getR2()))
print("R = "+str(exam.getR()))
