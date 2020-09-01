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

""" ======================  Variable Declaration ========================== """
order = 20
alpha = 0.0001
lr = 0.1

x = np.arange(0,70000)
""" ============================  Load Data =============================== """
data = sio.loadmat('project1.mat')
d = data['primary']
d = d.reshape(70000)
n = data['reference']
n = n.reshape(70000)

""" ===============================  NLMS ================================= """
w,e,N = LMS(n,d,lr,order)
soundfile.write('writeLMS.wav',e,21000)
winsound.PlaySound('writeLMS.wav',winsound.SND_FILENAME)
""" =======================  performance surface ========================== """
w1 = np.linspace(-4,4,100)
w2 = np.linspace(-4,4,100)
eps = np.zeros((100,100))
W1,W2 = np.meshgrid(w1,w2)
inputpower = np.dot(n,n)
for i in range(100):
    for j in range(100):
        wtemp = np.hstack((w1[i],w2[j]))
        y = np.dot(wtemp,N)
        eps[i,j] = np.sum((d - y)**2)/inputpower

plt.figure(1)
plt.contourf(W1,W2,eps,levels = 30)
#plt.contour(W1,W2,eps)
plt.colorbar()
plt.xlabel("w1")
plt.ylabel("w2")
plt.title("Performance Surface Contour")

""" ==========================  wight track =============================== """
plt.figure(2)
wtrack(w)
plt.title("Weight Tracks")

""" ==========================  learning curve =============================== """
steplr = 200
xlr = np.arange(0.001,0.2,0.001)
mse = np.zeros(steplr-1)
for i in range(steplr-1):
    mse[i] = LMSlr(n,d,i*0.001+0.001,order)

plt.figure(3)
plt.plot(xlr,mse)
plt.xlabel("Step Size")
plt.ylabel("MSE")
plt.title("Effect of Step Size")
plt.show()

plt.figure(4)
plt.plot(x,abs(e))
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.title("Learning Curve")

""" ==========================  frequency response =============================== """
efft = fft(e)
dfft = fft(d)

plt.figure(5)
plt.subplot(2,1,1)
plt.plot(x,abs(efft)**2/inputpower)
plt.xlabel("Number of Iterations")
plt.ylabel("Normalized Error")
plt.title("Error")

plt.subplot(2,1,2)
plt.plot(x,abs(dfft)**2/inputpower)
plt.xlabel("Number of Iterations")
plt.ylabel("Desired Signal")
plt.title("Desired Signal")

""" ==========================  SNR Improvement =============================== """
tepsnr = 20
xsnr = np.arange(0.01,0.2,0.01)
erle = np.zeros(stepsnr - 1)
for i in range(stepsnr - 1):
    w,e,N = LMS(n,d,i*0.01+0.01,order)
    erle[i] = 10*np.log(np.mean(d**2)/np.mean(e**2))

plt.figure(6)
plt.plot(xsnr,erle)
plt.xlabel("Step Size")
plt.ylabel("ERLE")
plt.title("SNR Improvement")

plt.show()

a=1