""" =======================  Import dependencies ========================== """
import numpy as np
import scipy
from scipy import signal
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft,ifft
import scipy.io as sio
import soundfile
import winsound

""" ======================  Function definitions ========================== """
def LMS(trainX, trainY, lr, order):
    step = len(trainX)
    w = np.zeros((order,step))
    X = np.zeros((order,len(trainX)))
    e = np.zeros(step)
    tempx = trainX
    for i in range(order):
        X[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:70000]

    for i in range(step-1):
        y = np.dot(w[:,i],X[:,i])
        e[i] = trainY[i] - y
        l = np.linalg.norm(X[:,i])
        w[:,i + 1] = w[:,i] + 2 * lr/(alpha + l) * e[i] * X[:,i]
    #wf = w[0:70000]
    return w,e,X

def LMSlr(trainX, trainY, lr, order):
    step = len(trainX)
    w = np.zeros((order,step))
    X = np.zeros((order,len(trainX)))
    e = np.zeros(step)
    tempx = trainX
    for i in range(order):
        X[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:70000]

    for i in range(step-1):
        y = np.dot(w[:,i],X[:,i])
        e[i] = trainY[i] - y
        l = np.linalg.norm(X[:,i])
        w[:,i + 1] = w[:,i] + 2 * lr/(alpha + l) * e[i] * X[:,i]
    mse = np.mean(np.power(e,2))
    return mse

def wtrack(w):
    row = len(w[:,0])
    column = len(w[0,:])
    x = np.arange(0,column)
    for i in range(row):
        s = 'w{n}'
        plt.plot(x, w[i,:], label = s.format(n = i))

    plt.xlabel("number of iterations")
    plt.ylabel("w")
    plt.legend()

def selfmul(x):
    order = len(x)
    res = np.zeros((order,order))
    for i in range(order):
        for j in range(order):
            res[i,j] = x[i]*x[j]
    return res

def RLS(x,d,order,a):

    itr = len(x)

    nr = np.eye(order)
    p = np.zeros((order,order))
    w = np.zeros(order)
    y = np.zeros(itr)
    e = np.zeros(itr)
    epower = np.zeros(itr)
    tempx = x
    X = np.zeros((order,len(x)))
    for i in range(order):
        #X[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:itr]
        X[i,:] = tempx

    for i in range(itr):
        y[i] = np.dot(w, X[:,i])
        e[i] = d[i] - y[i]
        z = np.dot(nr, X[:,i])
        q = np.dot(X[:,i], z)
        w = w + (e[i]*z)/(1 + q)
        nr = (1/a)*(nr - selfmul(z)/(a + q))

        epower[i] = e[i]**2/(inputpower + 1)

    mse = np.mean(epower)
    return mse,e
""" ======================  Variable Declaration ========================== """
order = 20
alpha = 0.0001
lr = 0.05

x = np.arange(0,70000)
""" ============================  Load Data =============================== """
data = sio.loadmat('project1.mat')
d = data['primary']
d = d.reshape(70000)
n = data['reference']
n = n.reshape(70000)
inputpower = np.dot(n,n)
""" ===============================  NLMS ================================= """
w,e,N = LMS(n,d,lr,order)
#soundfile.write('writeLMS.wav',e,21000)
#winsound.PlaySound('writeLMS.wav',winsound.SND_FILENAME)
""" =======================  order dicision ========================== """
'''stepor = 50
xor = np.arange(1,stepor)
erleor = np.zeros(stepor - 1)
for i in range(stepor - 1):
    w,e,N = LMS(n,d,0.01,i + 1)
    erleor[i] = 10*np.log(np.mean(d**2)/np.mean(e**2))

plt.plot(xor,erleor)
plt.xlabel("Filter Order Number")
plt.ylabel("ERLE")
plt.title("The Impact of Filter Order Number on ERLE")'''

""" ==========================  frequency response =============================== """
'''efft = fft(e)
dfft = fft(d)

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(x,abs(efft)**2/inputpower)
plt.xlabel("Number of Iterations")
plt.ylabel("Normalized Error")
plt.title("Error")

plt.subplot(2,1,2)
plt.plot(x,abs(dfft)**2/inputpower)
plt.xlabel("Number of Iterations")
plt.ylabel("Desired Signal")
plt.title("Desired Signal")'''

""" ==========================  SNR Improvement =============================== """
'''w,e2,N = LMS(n,d,lr,2)
w,e20,N = LMS(n,d,lr,20)

erle2 = 10*np.log(np.mean(d**2)/np.mean(e2**2))
erle20 = 10*np.log(np.mean(d**2)/np.mean(e20**2))

x1 = ['order = 2', 'order = 20']
y1 = [erle2,erle20]
plt.figure(2)
plt.bar(x1, y1)
plt.xlabel("filters of order")
plt.ylabel("ERLE")
plt.title("Impact of filter order on ERLE")'''


""" ==========================  The Impact of the Step Size on the Filter Perfomance =============================== """
'''steplr = 66
xlr = np.arange(0.001,0.066,0.001)
mse = np.zeros(steplr - 1)
for i in range(steplr - 1):
    mse[i] = LMSlr(n,d,i*0.001 + 0.001,order)

plt.figure(3)
plt.plot(xlr,mse)
plt.xlabel("Step Size")
plt.ylabel("MSE")
plt.title("Effect of Step Size")'''

""" ==========================  Misadjustment =============================== """
'''stepm = 66
stepsize = np.arange(0.001,1,0.01)
m = 2*inputpower*stepsize

plt.figure(4)
plt.plot(stepsize,m)
plt.xlabel("Step Size")
plt.ylabel("Misadjustment")
plt.title("Misadjustment")'''

""" ==========================  RLS and NLMS =============================== """
#mseRLS,eRLS = RLS(n,d,10,0.99)

'''stepor = 20
xor = np.arange(1,stepor)
erleorrls = np.zeros(stepor - 1)
erleorlms = np.zeros(stepor - 1)
for i in range(stepor - 1):
    erls,ee = RLS(n,d,i + 1,0.99)
    erleorrls[i] = 10*np.log(np.mean(d**2)/np.mean(erls**2))

    w,elms,N = LMS(n,d,0.01,i + 1)
    erleorlms[i] = 10*np.log(np.mean(d**2)/np.mean(elms**2))

plt.plot(xor,erleorrls,label = 'RLS')
plt.plot(xor,erleorlms,label = 'NLMS')
plt.xlabel("Filter Order Number")
plt.ylabel("ERLE")
plt.title("The Impact of Filter Order Number on ERLE between RLS and NLMS")'''

steplr = 10
xlr = np.arange(0.9,1,0.01)
mse = np.zeros(steplr)
for i in range(steplr):
    mse[i],e = RLS(n,d,10,i*0.01 + 0.9)

plt.figure(3)
plt.plot(xlr,mse)
plt.xlabel("Step Size")
plt.ylabel("MSE")
plt.title("Effect of Step Size")


plt.legend()
#soundfile.write('writeRLS.wav',eRLS,21000)
#winsound.PlaySound('writeRLS.wav',winsound.SND_FILENAME)
plt.show()
a=1