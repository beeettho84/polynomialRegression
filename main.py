import numpy as np
import math

prx = np.array([[108], [115], [106], [97], [95], [91], [97], [83], [83], [78], [54], [67], [56], [53], [61], [115], [81], [78], [30], [45], [99], [32], [25], [28], [90], [89]])
pry = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93]
seleccion = None
indice = None
B = None

class Dataset:
    x = np.array
    x2 = np.array
    x3 = np.array
    y = np.array
    sx = list()
    sx2 = list()
    sx3 = list()
    def __init__(self, inx, iny):
        self.sx = inx
        self.x = np.insert(inx, 0, 1, axis=1)
        unos = list()
        l1 = inx
        l2 = list()
        l3 = list()

        for i in range(len(inx)):
            unos.append(1)
            l2.append(l1[i][0]**2)
            l3.append(l1[i][0]**3)
        self.sx2 = l2
        self.sx3 = l3
        self.x2 = np.insert(self.x, 2, l2, axis=1)
        self.x3 = np.insert(self.x2, 2, l3, axis=1)
        self.y = iny

    def getX(self):
        return self.x

    def getXsimple(self):
        return self.sx

    def getX2simple(self):
        return self.sx2

    def getX3simple(self):
        return self.sx3

    def getX2(self):
        return self.x2

    def getX3(self):
        return self.x3

    def getY(self):
        return self.y


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
        Sx = DiscreteMath.SumX(self.data.getXsimple())  # obtenemos los valores parciales de las ecuaciones para facilitar el calculo
        Sy = DiscreteMath.SumY(self.data.getY())
        Sxy = DiscreteMath.SumXY(self.data.getXsimple(), self.data.getY())
        Sx2 = DiscreteMath.SumX2(self.data.getXsimple())
        Sy2 = DiscreteMath.SumY2(self.data.getY())
        self.B1 = (self.n * Sxy - (DiscreteMath.SumXSumY(self.data.getXsimple(), self.data.getY()))) / (self.n * Sx2 - (DiscreteMath.SumXSumX(self.data.getXsimple())))  # calculamos el valor de B1, necesario para obtener B0
        self.B0 = (Sy - (self.B1 * Sx)) / self.n  # calculamos B0
        Ssr = sum((yi - (self.B0 + self.B1 * xi)) ** 2 for xi, yi in zip(self.data.getX(), self.data.getY()))
        y_mean = Sy / self.n
        Sst = sum((yi - y_mean) ** 2 for yi in self.data.getY())
        self.r = ((self.n*Sxy)-(DiscreteMath.SumXSumY(self.data.getXsimple(), self.data.getY())))/math.sqrt((self.n*Sx2-(DiscreteMath.SumXSumX(self.data.getXsimple())))*(self.n*Sy2-(DiscreteMath.SumXSumX(self.data.getY()))))
        self.r2 = 1 - (Ssr / Sst)

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
    Bsl = list()
    Bsq = list()
    Bsc = list()
    r = [0,0,0]
    r2 = [0,0,0]
    n=0

    def __init__(self, inx, iny):
        if (len(inx) == len(iny)):
            self.n = len(inx)
            self.data = Dataset(inx, iny)
            self.CalculaBs()

    def CalculaBs(self):
            self.Bsl = self.RegresionLineal()
            self.getRLineal()
            self.Bsq = self.RegresionCuadratica()
            self.getRQuadratical()
            self.Bsc = self.RegresionCubica()
            self.getRCubical()
    def RegresionLineal(self):
        print("Lineal")
        Xt = np.transpose(self.data.getX())
        mr = np.dot(Xt,  self.data.getX())
        Xi = np.linalg.inv(mr)
        mr2 = (Xi @ Xt) @ self.data.getY()
        for i in range(len(mr2)):
            print("Beta ", str(i), ":", mr2[i])
        return mr2

    def RegresionCuadratica(self):
        print("Cuadratica")
        Xt = np.transpose(self.data.getX2())
        mr = np.dot(Xt,  self.data.getX2())
        Xi = np.linalg.inv(mr)
        mr2 = (Xi @ Xt) @ self.data.getY()
        for i in range(len(mr2)):
            print("Beta ", str(i), ":", mr2[i])
        return mr2

    def RegresionCubica(self):
        print("Cubica")
        Xt = np.transpose(self.data.getX3())
        mr = np.dot(Xt,  self.data.getX3())
        Xi = np.linalg.inv(mr)
        mr2 = (Xi @ Xt) @ self.data.getY()
        for i in range(len(mr2)):
            print("Beta ", str(i), ":", mr2[i])
        return mr2

    def predict(self, x):
        print("Prediccion lineal dado ", str(x), " : ", str(self.predictLineal(x)))
        print("Prediccion Cuadratica dado ", str(x), " : ", str(self.predictQuadratical(x)))
        print("Prediccion Cubica dado ", str(x), " : ", str(self.predictCubical(x)))

    def predictLineal(self, x):
        return self.Bsl[0] + (self.Bsl[1]*x)

    def predictQuadratical(self, x):
        return self.Bsq[0] + (self.Bsq[1]*x) + (self.Bsq[2]*(x**2))

    def predictCubical(self, x):
        return self.Bsc[0] + (self.Bsc[1]*x) + (self.Bsc[2]*(x**2)) + (self.Bsc[3]*(x**3))

    def getR2(self):
        return self.r2

    def getR(self):
        return self.r

    def getRLineal(self):
        Ssr = sum((yi - (self.predictLineal(xi))) ** 2 for xi, yi in zip(self.data.getXsimple(), self.data.getY()))
        y_mean = DiscreteMath.SumY(self.data.getY()) / self.n
        Sst = sum((yi - y_mean) ** 2 for yi in self.data.getY())
        self.r2[0] = 1 - (float(Ssr) / float(Sst))
        self.r[0] = math.sqrt(self.r2[0])
        print("R: ", str(self.r[0]))
        print("R2: ", str(self.r2[0]))

    def getRQuadratical(self):
        Ssr = sum((yi - (self.predictQuadratical(xi))) ** 2 for xi, yi in zip(self.data.getXsimple(), self.data.getY()))
        y_mean = DiscreteMath.SumY(self.data.getY()) / self.n
        Sst = sum((yi - y_mean) ** 2 for yi in self.data.getY())
        self.r2[1] = 1 - (float(Ssr) / float(Sst))
        self.r[1] = math.sqrt(self.r2[1])
        print("R: ", str(self.r[1]))
        print("R2: ", str(self.r2[1]))

    def getRCubical(self):
        Ssr = sum((yi - (self.predictCubical(xi))) ** 2 for xi, yi in zip(self.data.getXsimple(), self.data.getY()))
        y_mean = DiscreteMath.SumY(self.data.getY()) / self.n
        Sst = sum((yi - y_mean) ** 2 for yi in self.data.getY())
        self.r2[2] = 1 - (float(Ssr) / float(Sst))
        self.r[2] = math.sqrt(self.r2[2])
        print("R: ", str(self.r[2]))
        print("R2: ", str(self.r2[2]))

exam = prl(prx,pry)
exam.predict(24)
exam.predict(25)
exam.predict(27)
exam.predict(28)
exam.predict(29)

