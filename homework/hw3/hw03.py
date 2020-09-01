import numpy as np

import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import signal
from time import time
from sklearn.model_selection import train_test_split

def linaer_filter(y):
    y_delay = np.hstack((np.zeros((1)),y))
    t = -0.8*y + 0.7*y_delay[:5000]
    return t
def nonlinear_filter(t):
    return (t + 0.25*pow(t,2) + 0.11*pow(t,3))

def LMS(order,input,d,eta):
    w_init = np.random.uniform(-1,1,order)
    X = input
    E = []
    for i in range(1,order):
        delay = np.hstack((np.zeros((i)),input))
        delay = delay[:np.size(input)]
        X = np.vstack((X,delay))
    w = w_init
    W = w_init
    for i in range(0,np.size(input)):
        x = X[:,i]
        e = d[i]-w.T@x
        w = w - eta*(-2*e*x)/(.1+x.T@x)
        W = np.hstack((W,w))
        E = np.append(E,e**2)
    return W,E

def KLMS(x,d,eta,alpha,order):
    e = d[0]
    A = []
    E = []
    X = x
    
    for i in range(1,order):
        delay = np.hstack((np.zeros((i)),x))
        delay = delay[:np.size(x)]
        X = np.vstack((X,delay))
    beg = time()
    for i in range(1,np.size(x)):
        K = []
        A = np.append(A,e*eta)
        for j in range(i):
            k = 0
            for q in range(order):
                k = k + np.exp(-alpha*(X[q,j]-X[q,i])**2)
            K = np.append(K,k/order)
            # K = np.append(K,np.exp(-alpha*np.sum(np.abs(X[:,j]-X[:,i]))**2))
        e = d[i] - A@K.T
        E = np.append(E,e**2)
    end = time()
    print(end-beg)
    return E

def QKLMS(x,d,eta,alpha,order):
    e = d[0]
    epsilon = 1.5
    E = []
    X = x
    # C = [x[0]]
    for i in range(1,order):
        delay = np.hstack((np.zeros((i)),x))
        delay = delay[:np.size(x)]
        X = np.vstack((X,delay))
    A = [eta*d[0]]
    C = np.reshape(X[:,0],(order,1))
    begin = time()
    for i in range(1,np.size(x)):
        K = []
        D = []
        for j in range(np.size(C,1)):
            k = 0
            for q in range(order):
                k = k + np.exp(-alpha*(C[q,j]-X[q,i])**2)
            # K = np.append(K,np.exp(-alpha*(x[i]-C[j])**2))
            K = np.append(K,k/order)
            D = np.append(D,np.mean(np.abs(X[:,i]-C[:,j])))
            # D = np.append(D,x[i]-C[j])
        d_min = min(np.abs(D))
        index = np.argmin(D)
        e = d[i] - A@K.T
        if d_min < epsilon:
            A[index] = A[index] + eta*e
        else:
            A = np.append(A,eta*e)
            C = np.hstack((C,np.reshape(X[:,i],(order,1))))
            # C = np.append(C,x[i])
        E = np.append(E,e**2)
    end = time()

    print(end-begin)
    
    return E

def Ksize_train(x,d,eta,order):
    A = []
    E = []
    X = x
    e = d[0]
    mu = 10
    for i in range(1,order):
        delay = np.hstack((np.zeros((i)),x))
        delay = delay[:np.size(x)]
        X = np.vstack((X,delay))
    alpha = 10
    ALPHA = [alpha]
    E = [d[0]]
    beg = time()
    for i in range(1,np.size(x)):
        K = []
        A = np.append(A,e*eta)
        for j in range(i):
            k = 0
            for q in range(order):
                k = k + np.exp(-alpha*(X[q,j]-X[q,i])**2)
            K = np.append(K,k/order)
            # K = np.append(K,np.exp(-alpha*np.sum(np.abs(X[:,j]-X[:,i]))**2))
        e = d[i] - A@K.T
        E = np.append(E,e)
        alpha = alpha + 2*mu*eta*E[i]*E[i-1]*np.sum((X[:,i]-X[:,i-1])**2)*(k)/pow(alpha,3)
        ALPHA = np.append(ALPHA,alpha)
    end  = time()
    print(end-beg)
    plt.plot(ALPHA)
    plt.xlabel('Iteration')
    plt.ylabel('Kernel size')
    print('Kernel size equals to ',ALPHA[3999],'MSE equals to ',E[3999]**2)
    plt.show()
    return (ALPHA[3999])
    
def Ksize_test(x,d,order,eta,alpha):
    KLMS(x,d,eta,alpha,order)

def Problem1(y,q,x):
    plt.subplot(311)
    plt.plot(y,'r')
    plt.xlabel('Data Points')
    plt.ylabel('Amplitude')
    plt.title('y')
    plt.subplot(312)
    plt.plot(q,'g')
    plt.xlabel('Data Points')
    plt.ylabel('Amplitude')
    plt.title('q')
    plt.subplot(313)
    plt.plot(x,'b')
    plt.xlabel('Data Points')
    plt.ylabel('Amplitude')
    plt.title('x')
    plt.show()

y = norm.rvs(size=5000)
t = linaer_filter(y)
q = nonlinear_filter(t)
awgn = np.random.normal(0,1.5,5000)
x = q + awgn
d = np.hstack((y,np.zeros((2))))[2:]
order = 5
eta = .1
alpha = .01

x_train,x_test,d_train,d_test = train_test_split(x,d,test_size = 0.2)

E=KLMS(x,d,eta,alpha,order)
mseklms = np.mean(E)

a = 1
# Problem1(y,q,x)
# alpha = Ksize_train(x_train,d_train,eta,order)
# Ksize_test(x,d,order,eta,alpha)
# legend = ['LMS','KLMS','QKLMS']
# W,E = LMS(order,x,d,eta)
# plt.plot(E,'g')
# E = KLMS(x,d,eta,alpha,order)
# plt.plot(E,'r')
# E = QKLMS(x,d,eta,alpha,order)
# plt.plot(E,'b')
# plt.legend(legend)
# plt.show()