""" =======================  Import dependencies ========================== """
import numpy as np
import scipy
from scipy import signal
from scipy.stats import levy_stable
from scipy.stats import norm
from sklearn import metrics
import matplotlib.pyplot as plt
import math
import textwrap


"""
===============================================================================
===============================================================================
============================= Problem 1 =======================================
===============================================================================
===============================================================================
"""

""" ======================  Function definitions ========================== """
def autocorr(a, order, window):
    r = np.zeros(order)
    x = np.zeros((order,window))
    tempa = a[0:window]
    for i in range(order):
        x[i,:] = tempa
        tempa = np.insert(tempa,0,0)
        tempa = tempa[0:window]
    R = np.dot(x,x.T)
    return R

def crosscorr(a, b, order, window):
    r = np.zeros(order)
    x = np.zeros((order,window))
    tempa = a[0:window]
    tempb = b[0:window]
    for i in range(order):
        x[i,:] = tempa
        tempa = np.insert(tempa,0,0)
        tempa = tempa[0:window]
    R = np.dot(x,tempb.T)
    return R

def wsnr(x, d, order, window):
    r = autocorr(X, order, window)
    p = crosscorr(X, d, order, window)
    ropt = autocorr(X, order, 10000)
    popt = crosscorr(X, d, order, 10000)
    w = np.dot(np.linalg.inv(r), p)
    wopt = np.dot(np.linalg.inv(ropt), popt)
    wsnr = 10*(np.log(np.dot(wopt.T, wopt)/np.dot((wopt - w).T, (wopt - w))))
    return wsnr

def q3(X,noisepower):
    tout, tempX = signal.dlsim(Hz, X)
    noise_gaussian = np.random.normal(0, math.sqrt(noisepower), 10000)
    d = tempX.T + noise_gaussian
    d = d.flatten()

    r = autocorr(X, order, window)
    p = crosscorr(X, d, order, window)
    w = np.dot(np.linalg.inv(r), p)
    y = np.zeros(window)

    xtemp = np.zeros((order,window))
    tempa = X[0:window]
    for i in range(order):
        xtemp[i,:] = tempa
        tempa = np.insert(tempa,0,0)
        tempa = tempa[0:window]

    y = np.dot(w,xtemp)

    mse = metrics.mean_squared_error(d[0:window],y)
    wsnr3 = wsnr(X,d,order,window)
    return mse, wsnr3

class LMS:
    def __init__(self, vector_size):
        self.size   = vector_size
        self.weight = [0] * vector_size
        self.b      = 0
    def train(self, trainX, trainY, beta):
        step        = len(trainX)
        if step < self.size:
            return
        for i in range(self.size - 1, step):
            x       = trainX[i - self.size + 1: i + 1]
            value   = np.sum(self.weight * x)
            error   = trainY[i] - value
            self.weight += 2 * beta * error * x
        return self.weight
    def predict(self, input):
        predictY    = [np.nan] * (self.size - 1)
        for i in range(self.size - 1, len(input)):
            x       = input[i - self.size + 1: i + 1]
            predictY.append(np.sum(self.weight * x) + self.b)
        return predictY
    def evaluate(self, predictY, realY):
        loss = (predictY - realY) ** 2
        MSE = np.mean(loss[self.size - 1:])
        return MSE

""" ====================== Variable Declaration ========================== """
alpha = 1.8
beta = 0
gamma = 1
noisepower = 0.1
order = 5
window = 10000
""" ====================== Input Generation ========================== """
X = levy_stable.rvs(alpha,beta,size = 10000)
Hz = signal.TransferFunction([1,0,0,0,0,0,0,0,0,0,-1],[1,-1,0,0,0,0,0,0,0,0,0],dt = 1)
tout, tempX = signal.dlsim(Hz, X)
noise_gaussian = np.random.normal(0, math.sqrt(noisepower), 10000)
d = tempX.T + noise_gaussian
d = d.flatten()

""" ====================== i ========================== """
wsnr1 = wsnr(X,d,30,100)
wsnr2 = wsnr(X,d,30,500)
wsnr3 = wsnr(X,d,30,1000)

""" ====================== ii ========================== """
wsnr3 = wsnr(X,d,15,1000)
wsnr4 = wsnr(X[1000:len(X)],d,15,1000)
wsnr5 = wsnr(X[2000:len(X)],d,15,1000)
wsnr6 = wsnr(X[3000:len(X)],d,15,1000)

""" ====================== iii ========================== """
mse1,wsnr1 = q3(X,0.1)
mse2,wsnr2 = q3(X,0.3)
mse3,wsnr3 = q3(X,1.5)
""" ====================== iv ========================== """
ropt = autocorr(X, order, 10000)
popt = crosscorr(X, d, order, 10000)
wopt = np.dot(np.linalg.inv(ropt), popt)

model = LMS(order)
w4 = model.train(X[0:window], d[0:window], 0.001)
pre = model.predict(X[0:window])
mseLMS = model.evaluate(pre, d[0:window])
wsnrLMS = 10*(np.log(np.dot(wopt.T, wopt)/np.dot((wopt - w4).T, (wopt - w4))))

mseWF,wsnrWF = q3(X,0.1)
a=1