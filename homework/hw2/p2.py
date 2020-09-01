""" =======================  Import dependencies ========================== """
import numpy as np
import scipy
from scipy import signal
from sklearn import metrics
from scipy.io import wavfile
import matplotlib.pyplot as plt
import math
import datetime
""" ======================  Function definitions ========================== """
def selfmul(x):
    order = len(x)
    res = np.zeros((order,order))
    for i in range(order):
        for j in range(order):
            res[i,j] = x[i]*x[j]
    return res

def RLS(x,order,a):

    itr = len(x)

    #a = 2**(-1/len(x))
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
        e[i] = x[i] - y[i]
        z = np.dot(nr, X[:,i])
        q = np.dot(X[:,i], z)
        w = w + (e[i]*z)/(1 + q)
        nr = (1/a)*(nr - selfmul(z)/(a + q))

        inputpower = np.dot(X[:,i],X[:,i])/order
        epower[i] = e[i]**2/(inputpower + 1)

    epowermean = np.mean(epower)
    return w,epowermean,epower

def RLSq2(x,order,a):

    itr = len(x)

    #a = 2**(-1/len(x))
    nr = np.eye(order)
    p = np.zeros((order,order))
    w = np.zeros(order)
    y = np.zeros(itr)
    e = np.zeros(itr)
    epower = np.zeros(itr)
    tempx = x
    X = np.zeros((order,len(x)))
    W = np.zeros((order,itr))
    for i in range(order):
        #X[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:itr]
        X[i,:] = tempx

    for i in range(itr):
        y[i] = np.dot(w, X[:,i])
        e[i] = x[i] - y[i]
        z = np.dot(nr, X[:,i])
        q = np.dot(X[:,i], z)
        w = w + (e[i]*z)/(1 + q)
        nr = (1/a)*(nr - selfmul(z)/(a + q))
        W[:,i] = w

    return W

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

def APA(x,order,lr,k):
    itr = len(x) - k
    mu = 0.01
    #e = np.zeros(itr)
    epower = np.zeros(itr)
    norme = np.zeros(k)
    w = np.zeros(order)
    tempx = x
    X = np.zeros((order,len(x)))
    for i in range(order):
        #X[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:len(x)]
        X[i,:] = tempx


    for i in range(itr):
        u = X[:, i:i + k]
        g = np.dot(u.T,u) + mu*np.eye(k)
        a1 = x[i:i + k] - np.dot(u.T,w)
        a2 = np.linalg.inv(g)
        a3 = np.dot(u,a2)
        e = a1**2
        w = w + lr*np.dot(a3,a1)

        for j in range(k):
            inputpower = np.dot(u[:,j],u[:,j])
            norme[j] = e[j]/(inputpower + 1)
        epower[i] = np.mean(norme)
    epowermean = np.mean(epower)
    return w, epowermean,epower

def APAq4(x,order,lr,k):
    itr = len(x) - k
    mu = 0.01
    #e = np.zeros(itr)
    epower = np.zeros(itr)
    norme = np.zeros(k)
    w = np.zeros(order)
    tempx = x
    X = np.zeros((order,len(x)))
    W = np.zeros((order,itr))
    for i in range(order):
        #X[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:len(x)]
        X[i,:] = tempx


    for i in range(itr):
        u = X[:, i:i + k]
        g = np.dot(u.T,u) + mu*np.eye(k)
        a1 = x[i:i + k] - np.dot(u.T,w)
        a2 = np.linalg.inv(g)
        a3 = np.dot(u,a2)
        e = a1**2
        w = w + lr*np.dot(a3,a1)
        W[:,i] = w

    return W
""" ====================== Input Generation ========================== """
sample_rate, sig = wavfile.read('we were away a year ago.WAV')
order = 15
sigpower = np.sum(np.power(sig,2))
step = len(sig)
x = np.arange(0,step)

""" ============================ i ================================= """
'''step1 = 20
x = np.arange(0.9, 1, 0.005)
e0 = np.zeros(step1)
for i in range(step1):
    w0,e0[i],epower0 = RLS(sig, order, i/200 + 0.9)

plt.figure()
plt.plot(x, e0, marker='o')
plt.xlabel("forgetting factor")
plt.ylabel("mean of normalized error power")
plt.title("the effect of the forgetting factor")
plt.show()


w1,e1,epower1 = RLS(sig,6,0.99)
w2,e2,epower2 = RLS(sig,15,0.99)
x1 = ['order = 6', 'order = 15']
y1 = [e1, e2]
plt.figure()
plt.bar(x1,y1)
plt.xlabel("filters of order")
plt.ylabel("mean of normalized error power")
plt.show()

w1,e1,epower1 = RLS(sig, order, 0.1)
w2,e2,epower2 = RLS(sig, order, 0.3)
w3,e3,epower3 = RLS(sig, order, 0.5)
w4,e4,epower4 = RLS(sig, order, 0.7)
w5,e5,epower5 = RLS(sig, order, 0.9)
w6,e6,epower6 = RLS(sig, order, 0.99)

x = np.arange(0,step)
plt.figure(figsize = (16,12))

plt.subplot(3,2,1)
plt.plot(x,abs(epower1))
plt.xlabel("number of iterations")
plt.ylabel("normalized error power")
plt.title("forgetting factor = 0.1")

plt.subplot(3,2,2)
plt.plot(x,abs(epower2))
plt.xlabel("number of iterations")
plt.ylabel("normalized error power")
plt.title("forgetting factor = 0.3")

plt.subplot(3,2,3)
plt.plot(x,abs(epower3))
plt.xlabel("number of iterations")
plt.ylabel("normalized error power")
plt.title("forgetting factor = 0.5")

plt.subplot(3,2,4)
plt.plot(x,abs(epower4))
plt.xlabel("number of iterations")
plt.ylabel("normalized error power")
plt.title("forgetting factor = 0.7")

plt.subplot(3,2,5)
plt.plot(x,abs(epower5))
plt.xlabel("number of iterations")
plt.ylabel("normalized error power")
plt.title("forgetting factor = 0.9")

plt.subplot(3,2,6)
plt.plot(x,abs(epower6))
plt.xlabel("number of iterations")
plt.ylabel("normalized error power")
plt.title("forgetting factor = 0.99")

plt.tight_layout(pad=0.4, w_pad=4.0, h_pad=3.0)
plt.show()'''
""" ============================ ii ================================= """
'''W = RLSq2(sig,15,0.99)
wtrack(W)
plt.show()'''

""" ============================ iii ================================= """
'''w1,e1,epower1 = RLS(sig, 6, 0.99)
w2,e2,epower2 = RLS(sig, 15, 0.99)

plt.figure()

plt.subplot(2,1,1)
plt.plot(x,abs(epower1))
plt.xlabel("number of iterations")
plt.ylabel("normalized error power")
plt.title("filters of order 6")

plt.subplot(2,1,2)
plt.plot(x,abs(epower2))
plt.xlabel("number of iterations")
plt.ylabel("normalized error power")
plt.title("filters of order 15")
plt.show()'''

""" ============================ iv ================================= """
'''step = 10
epowermeanAPA = np.zeros(step)
x0 = np.arange(0.15,0.25,0.01)
for i in range(step):
    wAPA, epowermeanAPA[i], epowerAPA = APA(sig, 15, i*0.01 + 0.15, 100)
#wRLS,epowermeanRLS,epowerRLS = RLS(sig,15,0.98)

plt.plot(x0,epowermeanAPA)
plt.xlabel("learning rate")
plt.ylabel("normalized error power")
plt.show()

W = APAq4(sig,15,0.17,100)
wtrack(W)
plt.show()

w1,e1,epower1 = APA(sig,6,0.17,100)
w2,e2,epower2 = APA(sig,15,0.17,100)
x1 = ['order = 6', 'order = 15']
y1 = [e1, e2]
plt.figure()
plt.bar(x1,y1)
plt.xlabel("filters of order")
plt.ylabel("normalized error power")
plt.show()'''

'''wRLS,epowermeanRLS,epowerRLS = RLS(sig,15,0.98)
wAPA, epowermeanAPA, epowerAPA = APA(sig, 15, 0.17,100)
#wAPA1, epowermeanAPA1, epowerAPA1 = APA(sig, 15, 0.1,200)
plt.figure(1)

x1 = ['RLS', 'APA']
y1 = [epowermeanRLS, epowermeanAPA]
plt.bar(x1,y1)
plt.xlabel("method")
plt.ylabel("mean of normalized error power")

plt.figure(2)
x2 = np.arange(0,len(x)-100)
plt.subplot(2,1,1)
plt.plot(x,abs(epowerRLS))
plt.xlabel("number of iterations")
plt.ylabel("normalized error power")
plt.title("RLS")

plt.subplot(2,1,2)
plt.plot(x2,abs(epowerAPA))
plt.xlabel("number of iterations")
plt.ylabel("normalized error power")
plt.title("APA")

plt.show()
'''

""" ============================ v ================================= """
step = 10
epowermeanAPA = np.zeros(step)
time = np.zeros(step)
x0 = np.arange(100,1100,100)
for i in range(step):
    starttime = datetime.datetime.now()
    wAPA, epowermeanAPA[i], epowerAPA = APA(sig, 15, 0.17, i*100 + 100)
    endtime = datetime.datetime.now()
    time[i] = (endtime - starttime).seconds


'''plt.plot(x0,epowermeanAPA)
plt.xlabel("number of samples")
plt.ylabel("mean of normalized error power")'''
plt.plot(x0,time)
plt.xlabel("number of samples")
plt.ylabel("operation time")
plt.show()
a=1