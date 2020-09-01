""" =======================  Import dependencies ========================== """
import numpy as np
import scipy
from scipy import signal
from scipy.stats import levy_stable
from sklearn import metrics
import matplotlib.pyplot as plt
import math

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

def wsnr(w, order):
    if order <=10:
        wopt = np.ones(order)
    else:
        wopt1 = np.ones(10)
        wopt2 = np.zeros(order - 10)
        wopt = np.hstack((wopt1, wopt2))
    wsnr = 10*(np.log(np.dot(wopt.T, wopt)/np.dot((wopt - w).T, (wopt - w))))
    return wsnr

def changenoise(X,noisepower):
    tout, tempX = signal.dlsim(Hz, X)
    noise_gaussian = np.random.normal(0, math.sqrt(noisepower), 10000)
    d = tempX.T + noise_gaussian
    d = d.flatten()

    wsnr,e,W = RLS(x,d,order,10000)


    return wsnr,e,W

def q3(x,d):
    wsnr,e,W = RLS(x,d,order,10000)

    return wsnr,e,W
    
def LMS(trainX, trainY, beta, order):
    w = np.zeros(order)
    step = len(trainX)
    X = np.zeros((order,len(trainX)))
    e = np.zeros(step)
    tempx = trainX
    for i in range(order):
        X[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:10000]

    for i in range(step):
        y = np.dot(w,X[:,i])
        e[i] = trainY[i] - y
        w = w + 2 * beta * e[i] * X[:,i]
    return w,e

def selfmul(x):
    order = len(x)
    res = np.zeros((order,order))
    for i in range(order):
        for j in range(order):
            res[i,j] = x[i]*x[j]
    return res

def RLS(x,d,order,itr):
    a = 2**(-1/len(x))
    nr = np.eye(order)
    p = np.zeros((order,order))
    w = np.zeros(order)
    y = np.zeros(itr)
    e = np.zeros(itr)
    tempx = x
    X = np.zeros((order,len(x)))
    W = np.zeros((order,itr))
    for i in range(order):
        X[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:10000]
        #X[i,:] = tempx

    for i in range(itr):
        y[i] = np.dot(w, X[:,i])
        e[i] = d[i] - y[i]
        z = np.dot(nr, X[:,i])
        q = np.dot(X[:,i], z)
        w = w + (e[i]*z)/(1 + q)
        nr = (1/a)*(nr - selfmul(z)/(a + q))
        W[:,i] = w

    wsnrrls = wsnr(w,order)
    return wsnrrls,e,W

def wiener(X,d):
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
    wsnr2 = wsnr(w,order)
    return mse, wsnr2

def wtrack(w):
    row = len(w[:,0])
    column = len(w[0,:])
    for i in range(row):
        s = 'w{n}'
        plt.plot(x, w[i,:], label = s.format(n = i))
    plt.legend()
""" ====================== Variable Declaration ========================== """
alpha = 1.5
beta = 0
gamma = 1
noisepower = 0.1
order = 15
window = 1000

""" ====================== Input Generation ========================== """
x = levy_stable.rvs(alpha,beta,size = 10000)
Hz = signal.TransferFunction([1,0,0,0,0,0,0,0,0,0,-1],[1,-1,0,0,0,0,0,0,0,0,0],dt = 1)
tout, tempX = signal.dlsim(Hz, x)
noise_gaussian = np.random.normal(0, math.sqrt(noisepower), 10000)
d = tempX.T + noise_gaussian
d = d.flatten()

""" ============================ i ================================= """
'''wsnr1,e1,W1 = RLS(x,d,5,10000)
wsnr2,e2,W2 = RLS(x,d,15,10000)
wsnr3,e3,W3 = RLS(x,d,30,10000)

x = np.arange(0,10000)
plt.figure(figsize = (16,12))

plt.subplot(3,2,1)
plt.plot(x,abs(e1))
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.title("order = 5")

plt.subplot(3,2,2)
wtrack(W1)
plt.xlabel("number of iterations")
plt.ylabel("w")
plt.title("order = 5")

plt.subplot(3,2,3)
plt.plot(x,abs(e2))
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.title("order = 15")

plt.subplot(3,2,4)
wtrack(W2)
plt.xlabel("number of iterations")
plt.ylabel("w")
plt.title("order = 15")

plt.subplot(3,2,5)
plt.plot(x,abs(e3))
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.title("order = 30")

plt.subplot(3,2,6)
wtrack(W3)
plt.xlabel("number of iterations")
plt.ylabel("w")
plt.title("order = 30")

plt.tight_layout(pad=0.4, w_pad=4.0, h_pad=3.0)
plt.show()

x1 = [5,15,30]
y1 = [wsnr1,wsnr2,wsnr3]
plt.plot(x1,y1)
plt.xlabel("filters of order")
plt.ylabel("WSNR")
plt.show()'''
""" ============================ ii ================================= """
'''wsnr4,e4,W4 = changenoise(x, 0.1)
wsnr4,e5,W5 = changenoise(x, 0.3)
wsnr4,e6,W6 = changenoise(x, 1.5)

x = np.arange(0,10000)
plt.figure(figsize = (16,12))

plt.subplot(3,2,1)
plt.plot(x,abs(e4))
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.title("noise = 0.1")

plt.subplot(3,2,2)
wtrack(W4)
plt.xlabel("number of iterations")
plt.ylabel("w")
plt.title("noise = 0.1")

plt.subplot(3,2,3)
plt.plot(x,abs(e5))
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.title("noise = 0.3")

plt.subplot(3,2,4)
wtrack(W5)
plt.xlabel("number of iterations")
plt.ylabel("w")
plt.title("noise = 0.3")

plt.subplot(3,2,5)
plt.plot(x,abs(e6))
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.title("noise = 1.5")

plt.subplot(3,2,6)
wtrack(W6)
plt.xlabel("number of iterations")
plt.ylabel("w")
plt.title("noise = 1.5")

plt.tight_layout(pad=0.4, w_pad=4.0, h_pad=3.0)
plt.show()

x2 = [0.1,0.3,1.5]
y2 = [wsnr4,wsnr5,wsnr6]
plt.plot(x2,y2)
plt.xlabel("noise power")
plt.ylabel("WSNR")
plt.show()'''
""" ============================ iii ================================= """   

wLMS, eLMS = LMS(x, d, 0.001, order)
wsnrLMS = wsnr(wLMS,order)

mseWF,wsnrWF = wiener(x,d)
wsnrRLS,eRLS,WRLS = RLS(x,d,order,10000)

x = np.arange(0,10000)
plt.figure()

plt.subplot(2,1,1)
plt.plot(x,abs(eLMS))
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.title("LMS")

plt.subplot(2,1,2)
plt.plot(x,abs(eRLS))
plt.xlabel("number of iterations")
plt.ylabel("error")
plt.title("RLS")

plt.show()

x3 = ["LMS","Wiener","RLS"]
y3 = [wsnrLMS,wsnrWF,wsnrRLS]
plt.bar(x3,y3)
plt.xlabel("method")
plt.ylabel("WSNR")
plt.show()

c=1