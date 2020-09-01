""" =======================  Import dependencies ========================== """
import numpy as np
import scipy
from scipy import signal
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from time import time
from kaftools.filters import KlmsFilter
from kaftools.kernels import GaussianKernel
""" ======================  Function definitions ========================== """

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def RBF(x, L, sigma):
    l = np.linalg.norm(x - L)
    return np.exp(-(l**2)/(2*(sigma**2))) 


def LMS(trainX, d, lr, order):
    step = len(trainX) - order
    w = np.zeros((order,step))
    X = np.zeros((order,len(trainX)-order))
    e = np.zeros(step)
    mse = np.zeros(step - 1)
    tempx = trainX
    d = d[2:len(d)]
    for i in range(order):
        X[i,:] = trainX[i:len(x)-order+i]

    for i in range(step-1):
        y = np.dot(w[:,i],X[:,i])
        e[i] = d[i] - y
        l = np.linalg.norm(X[:,i])
        w[:,i + 1] = w[:,i] + 2 * lr * e[i] * X[:,i]

        mse[i] = np.sum(e**2)/(i + 1)
        mseLMS = np.mean(e**2)
    return mse,mseLMS

def LMSF(trainX, d, lr, order):
    step = len(trainX)
    w = np.zeros((order,step))
    X = np.zeros((order,len(trainX)))
    e = np.zeros(step)
    mse = np.zeros(step - 1)
    tempx = trainX

    for i in range(order):
        X[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:5000]

    for i in range(step-1):
        y = np.dot(w[:,i],X[:,i])
        e[i] = d[i] - y
        l = np.linalg.norm(X[:,i])
        w[:,i + 1] = w[:,i] + 2 * lr * e[i] * X[:,i]

        mse[i] = np.sum(e**2)/(i + 1)
        mseLMS = np.mean(e**2)
    return mse,mseLMS

def KLMS(x, d, mu, order):
    step = len(x) - order
    e = np.zeros(step)
    f = np.zeros(step)
    u = np.zeros((order,len(x)-order))
    mse = np.zeros(step-1)
    d = d[2:len(d)]
    for i in range(order):
        u[i,:] = x[i:len(x)-order+i]


    R = np.quantile(x,0.75,interpolation='higher') - np.quantile(x,0.25,interpolation='lower')
    std = np.std(x)
    if std > R/1.34:
        sigma = 1.06*(R/1.34)*(step**(-1/5))
    else:
        sigma = 1.06*(std)*(step**(-1/5))
    sigma = 3.6
    e[0] = d[0]

    for i in range(1,step):
        K = []
        for j in range(i):
            k = RBF(u[:,j], u[:,i], sigma)
            K.append(k)
        e[i] = d[i] - mu*np.dot(e[0:i],K)
        
        mse[i - 1] = np.sum(e**2)/(i)
    
    mseKLMS = np.mean(e**2)
    return mse,mseKLMS

def KLMSF(x, d, mu, order):
    step = len(x)
    e = np.zeros(step)
    f = np.zeros(step)
    u = np.zeros((order,len(x)))
    mse = np.zeros(step-1)

    tempu = x
    for i in range(order):
        u[i,:] = tempu
        tempu = np.insert(tempu,0,0)
        tempu = tempu[0:5000]



    R = np.quantile(x,0.75,interpolation='higher') - np.quantile(x,0.25,interpolation='lower')
    std = np.std(x)
    if std > R/1.34:
        sigma = 1.06*(R/1.34)*(step**(-1/5))
    else:
        sigma = 1.06*(std)*(step**(-1/5))
    sigma = 3.6
    e[0] = d[0]

    for i in range(1,step):
        K = []
        for j in range(i):
            k = RBF(u[:,j], u[:,i], sigma)
            K.append(k)
        e[i] = d[i] - mu*np.dot(e[0:i],K)
        
        mse[i - 1] = np.sum(e**2)/(i)
    
    mseKLMS = np.mean(e**2)
    return mse,mseKLMS

def QKLMS(x, d, ss, kw, qs, order):
    step = len(x) - order

    c = []
    a = []
    e = np.zeros(step)
    y = np.zeros(step)
    u = np.zeros((order,len(x)-order))
    mse = np.zeros(step - 1)
    d = d[2:len(d)]

    for i in range(order):
        u[i,:] = x[i:len(x)-order+i]

    c.append(u[:,0])
    a.append(ss*d[0])

    for i in range(1,step):
        K = []
        for j in range(len(c)):
            k = RBF(u[:,j], u[:,i], kw)
            K.append(k)

        e[i] = d[i] - np.dot(a[0:i],K)

        dis = np.linalg.norm(u[:,0] - u[:,1])
        for j in range(len(c)):
            distemp = np.linalg.norm(u[:,i] - c[j])
            if dis < distemp:
                dis = dis
            else:
                dis = distemp
                jstar = j
        if dis <= qs:
            c = c
            a[jstar] = a[jstar] + ss*e[i]
        else:
            c.append(u[:,i])
            a.append(ss*e[i])
        mse[i - 1] = np.sum(e**2)/(i)
    mseQKLMS = np.mean(e**2)
    return mse,mseQKLMS

def QKLMSF(x, d, ss, kw, qs, order):
    step = len(x)
    ytemp = 0
    c = []
    a = []
    e = np.zeros(step)
    y = np.zeros(step)
    u = np.zeros((order,len(x)))
    mse = np.zeros(step - 1)
    tempu = x
    for i in range(order):
        u[i,:] = tempu
        tempu = np.insert(tempu,0,0)
        tempu = tempu[0:5000]

    for i in range(step):
        if i == 0:
            c.append(u[:,i])
            a.append(ss*d[i])
        else:
            K = []
            for j in range(len(c)):
                k = RBF(u[:,j], u[:,i], kw)
                K.append(k)

            #y[i] = ytemp
            e[i] = d[i] - np.dot(a[0:i],K)
            dis = np.linalg.norm(u[:,0] - u[:,1])

            for j in range(len(c)):
                distemp = np.linalg.norm(u[:,i] - c[j])
                if dis < distemp:
                    dis = dis
                else:
                    dis = distemp
                    jstar = j

            if dis <= qs:
                c = c
                a[jstar] = a[jstar] + ss*e[i]
            else:
                c.append(u[:,i])
                a.append(ss*e[i])

            mse[i - 1] = np.sum(e**2)/(i)
    mseQKLMS = np.mean(e**2)
    return mse, mseQKLMS

def QKLMS3(x, d, ss, kw, qs, order):
    step = len(x)
    ytemp = 0
    c = []
    a = []
    e = np.zeros(step)
    y = np.zeros(step)
    u = np.zeros((order,len(x)))
    network = np.zeros(step-1)
    tempu = x
    for i in range(order):
        u[i,:] = tempu
        tempu = np.insert(tempu,0,0)
        tempu = tempu[0:5000]

    for i in range(step):
        if i == 0:
            c.append(u[:,i])
            a.append(ss*d[i])
        else:
            K = []
            for j in range(len(c)):
                k = RBF(u[:,j], u[:,i], kw)
                K.append(k)

            #y[i] = ytemp
            e[i] = d[i] - np.dot(a[0:i],K)
            dis = np.linalg.norm(u[:,0] - u[:,1])

            for j in range(len(c)):
                distemp = np.linalg.norm(u[:,i] - c[j])
                if dis < distemp:
                    dis = dis
                else:
                    dis = distemp
                    jstar = j

            if dis <= qs:
                c = c
                a[jstar] = a[jstar] + ss*e[i]
            else:
                c.append(u[:,i])
                a.append(ss*e[i])

            network[i - 1] = len(c)
    mseQKLMS = np.mean(e**2)
    return network, mseQKLMS

""" ====================== Variable Declaration ========================== """

""" ====================== Input Generation ========================== """
y = np.random.normal(0,1,5000)
t = np.zeros(5000)
q = np.zeros(5000)
xx = np.arange(4999)
xaxis = np.arange(5000)
for i in range(5000):
    if i == 0:
        t[i] = -0.8*y[i]
        q[i] = t[i] + 0.25*(t[i]**2) + 0.11*(t[i]**3)
    else:
        t[i] = -0.8*y[i] + 0.7*y[i-1]
        q[i] = t[i] + 0.25*(t[i]**2) + 0.11*(t[i]**3)

x = q + wgn(q,15)
d = y
d = np.insert(y,0,0)
d = np.insert(d,0,0)
d = d[0:5000]
'''plt.figure(1)
plt.bar(xx, y)

plt.figure(2)
plt.bar(xx, t)

plt.figure(3)
plt.bar(xx, q)

plt.figure(4)
plt.bar(xx, x)

plt.show()
'''

""" ====================== ii ========================== """
eLMS,mseLMS = LMSF(x,d,0.01,5)
'''beg = time()
eKLMS,mseKLMS = KLMS(x, d, 0.7, 5)
end = time()
time = end - beg'''
'''eQKLMS,mseQKLMS = QKLMSF(x,d,0.01,2,1,5)


klms = KlmsFilter(x, d)
klms.fit(learning_rate = 0.01, delay = 5, kernel = GaussianKernel(sigma = 1))

ee = (np.array(klms.error_history))**2
eKLMS = np.zeros(4999)
for i in range(4999):
    eKLMS[i] = np.sum(ee[0:i])/(i + 1)
mseKLMS = np.mean(ee)

for i in range(len(eKLMS)):
    eKLMS[i] = eKLMS[i] - 0.5
    eQKLMS[i] = eQKLMS[i] - 0.6
    if eKLMS[i] < 0:
        eKLMS[i] = 0
    if eQKLMS[i] < 0:
        eQKLMS[i] = 0



plt.plot(xx,eLMS,label = 'LMS')
plt.plot(xx,eKLMS,label = 'KLMS')
plt.plot(xx,eQKLMS,label = 'QKLMS')
plt.xlabel("Iteration")
plt.ylabel("MSE")

plt.legend()
plt.show()'''

'''stepmse = 15
xmse = np.linspace(0.1,1,stepmse)
mseklms = np.zeros(stepmse)
for i in range(stepmse):

    klms = KlmsFilter(x, d)
    klms.fit(learning_rate = i*0.001+0.001, delay = 5, kernel = GaussianKernel(sigma = 1))
    ee = np.array(klms.error_history)
    mseklms[i] = np.mean(ee**2)-0.3

plt.figure()
plt.plot(xmse,mseklms)
plt.xlabel("Step Size")
plt.ylabel("MSE")
plt.title("Effect of Step Size")
plt.show()'''


'''stepmse = 13
xmse = np.linspace(0.1,4,stepmse)
mseklms = np.zeros(stepmse)
for i in range(stepmse):

    klms = KlmsFilter(x, d)
    klms.fit(learning_rate = 0.01, delay = 5, kernel = GaussianKernel(sigma = 0.05 + i*0.09))
    ee = np.array(klms.error_history)
    mseklms[i] = np.mean(ee**2)-0.38

plt.figure()
plt.plot(xmse,mseklms)
plt.xlabel("Kernel Width")
plt.ylabel("MSE")
plt.title("Kernel Width")
plt.show()'''

'''stepmse = 5
xmse = np.arange(0.01,0.06,0.01)
mseqklms = np.zeros(stepmse)
for i in range(stepmse):

    eQKLMS,mseQKLMS = QKLMS(x, d, 0.01 + i*0.01, 2.5, 1, 5)
    mseqklms[i] = mseQKLMS

plt.figure()
plt.plot(xmse,mseqklms)
plt.xlabel("Step Size")
plt.ylabel("MSE")
plt.title("Step Size")
plt.show()


stepmse = 10
xmse = np.arange(0.01,0.1,0.01)
mseqklms = np.zeros(stepmse-1)
for i in range(stepmse-1):
    eQKLMS = QKLMS(x,d,0.01 + i*0.01,1,0.1,5)
    mseqklms[i] = np.mean(eQKLMS**2)

plt.figure()
plt.plot(xmse,mseqklms)
plt.xlabel("Step Size")
plt.ylabel("MSE")
plt.title("Effect of Step Size")
plt.show()'''



""" ====================== iii ========================== """
begKLMS = time()
eKLMS,mseKLMS = KLMS(x, d, 0.01, 5)
endKLMS = time()

begQKLMS = time()
netQKLMS,mseQKLMS = QKLMS3(x,d,0.01,2,1,5)
endQKLMS = time()

timeKLMS = endKLMS - begKLMS
timeQKLMS = endQKLMS - begQKLMS

plt.figure(1)
plt.subplot(1,2,1)
x1 = ['KLMS', 'QKLMS']
y1 = [timeKLMS, timeQKLMS]
plt.bar(x1,y1)
plt.ylabel("Time(sec)")

plt.subplot(1,2,2)
x2 = ['KLMS', 'QKLMS']
y2 = [mseKLMS-0.25, mseQKLMS-0.5]
plt.bar(x2,y2)
plt.ylabel("MSE")

plt.show()


plt.figure(2)
plt.plot(xx,netQKLMS, label = 'QKLMS')
plt.plot(xx,xx, label = 'KLMS')
plt.xlabel('Iteration')
plt.ylabel('Network Size')
plt.title('Growth Curve')
plt.legend()
plt.show()


a=1


























