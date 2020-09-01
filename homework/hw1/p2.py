""" =======================  Import dependencies ========================== """
import numpy as np
import scipy
from scipy import signal
from scipy.stats import norm
from scipy.io import wavfile
from sklearn import metrics
import matplotlib.pyplot as plt
import math

""" ======================  Function definitions ========================== """
def autocorr(a, order, window):
    x = np.zeros((order,window))
    for i in range(order):
        x[i,:] = a[i:i + window]
    R = np.dot(x,x.T)
    return R

def crosscorr(a, b, order, window):
    x = np.zeros((order,window))
    for i in range(order):
        x[i,:] = a[i:i + window]
    R = np.dot(x,b)
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

def epower(order, window):
    step = len(sig) - window - order
    w = np.zeros((step, order))
    wmean = np.zeros(step)
    for i in range(step):
        r = autocorr(sig[i:i + window + order], order, window)
        p = crosscorr(sig[i:i + window + order], sig[i + order:i + window + order], order, window)

        w[i,:] = np.dot(np.linalg.inv(r), p)
        wmean[i] = np.mean(np.abs(w[i,:]))

    y = np.zeros(step)
    for i in range(step):
        y[i] = np.dot(w[i,:],(sig[i:i + order]).T)

    e = sig[order:order + step] - y
    epower = np.mean(np.power(e,2))/sigpower
    return epower

class LMS:
    def __init__(self, vector_size):
        self.size   = vector_size
        self.weight = [0] * vector_size
        self.b      = 0
    def train(self, trainX, trainY, beta):
        step        = len(trainX)
        wmean = np.zeros(step)
        error1 = np.zeros(step)
        if step < self.size:
            return
        for i in range(self.size - 1, step):
            x       = trainX[i - self.size + 1: i + 1]
            value   = np.sum(self.weight * x)
            error   = trainY[i] - value
            self.weight += (2 * beta * error * x)/np.sum(np.power(trainX,2))
            wmean[i] = np.mean(np.abs(self.weight))
            error1[i] = error
        return error1
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

""" ====================== Input Generation ========================== """
sample_rate, sig = wavfile.read('we were away a year ago.WAV')
order = 6
window = 100
step = len(sig) - window - order
x = np.arange(0,step/10000,0.0001)
sigpower = np.mean(np.power(sig,2))
""" ====================== ii ========================== """
w = np.zeros((step, order))
wmean = np.zeros(step)
for i in range(step):
    r = autocorr(sig[i:i + window + order], order, window)
    p = crosscorr(sig[i:i + window + order], sig[i + order:i + window + order], order, window)

    w[i,:] = np.dot(np.linalg.inv(r), p)
    wmean[i] = np.mean(np.abs(w[i,:]))

y = np.zeros(step)
for i in range(step):
    y[i] = np.dot(w[i,:],(sig[i:i + order]).T)

e = sig[order:order + step] - y
epowerWF = np.mean(np.power(e,2))/sigpower
'''epower1 = epower(6,100)
epower2 = epower(6,200)
epower3 = epower(6,500)
epower4 = epower(15,100)
epower5 = epower(15,200)
epower6 = epower(15,500)'''
'''plt.figure(1)
plt.xlabel('t(secs)')
plt.ylabel('w')
plt.plot(x,wmean)

plt.figure(2)
plt.xlabel('t(secs)')
plt.ylabel('error')
plt.plot(x,np.abs(e)/10000)
plt.show()'''

""" ====================== iv ========================== """
model = LMS(order)
eLMS = model.train(sig[0:step], sig[order:order+step], 0.001)
pre = model.predict(sig[0:step])
mseLMS = model.evaluate(pre, sig[order:order+step])
xLMS = np.arange(0,18070/10000,0.0001)
epowerLMS = np.mean(np.power(eLMS,2))/(sigpower*100)
plt.figure()
plt.xlabel('t(secs)')
plt.ylabel('w')
plt.plot(x,w4)
plt.show()
a=1