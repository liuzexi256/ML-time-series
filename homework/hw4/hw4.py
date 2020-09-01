""" =======================  Import dependencies ========================== """
import numpy as np
import scipy
from scipy import signal
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import mlab
import math 
from scipy.fftpack import fft,ifft
from scipy import signal
import scipy.io as sio
import soundfile
import winsound

""" ======================  Function definitions ========================== """
def generate_sinusoid(N, A, f0, fs, phi):
    '''
    N(int) : number of samples
    A(float) : amplitude
    f0(float): frequency in Hz
    fs(float): sample rate
    phi(float): initial phase
    
    return 
    x (numpy array): sinusoid signal which lenght is M
    '''
    
    T = 1/fs
    n = np.arange(N)    # [0,1,..., N-1]
    x = A * np.sin( 2*f0*np.pi*n*T + phi )
    
    return x

def generate_sinusoid_2(t, A, f0, fs, phi):
    '''
    t  (float) : time length of the generated sequence
    A  (float) : amplitude
    f0 (float) : frequency
    fs (float) : sample rate
    phi(float) : initial phase
    
    returns
    x (numpy array): sinusoid signal sequence
    '''
    
    T = 1.0/fs
    N = t / T
    
    return generate_sinusoid(N, A, f0, fs, phi)

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

def wiener(X, d, order, window):
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

    mse = metrics.mean_squared_error(d, y)
    return y, mse

def LMS(x, d, lr, order):
    step = len(x)
    w = np.zeros((order,step))
    X = np.zeros((order,step))
    e = np.zeros(step)
    y = np.zeros(step)
    allmse = np.zeros(step)
    tempx = x
    for i in range(order):
        X[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:20000]

    for i in range(step - 1):
        y[i] = np.dot(w[:,i],X[:,i])
        e[i] = d[i] - y[i]
        #l = np.linalg.norm(X[:,i])
        w[:,i + 1] = w[:,i] + 2 * lr * e[i] * X[:,i]
        allmse[i + 1] = np.sum(e**2)/(i + 1)
    mse = np.mean(np.power(e,2))
    return y, mse

def RBF(x, L, sigma):
    l = np.linalg.norm(x - L)
    return math.exp(-(l**2)/(2*(sigma**2))) 

def KLMS(x, d, lr, order, sigma):
    step = len(x)
    e = np.zeros(step)
    y = np.zeros(step)
    f = np.zeros(step)
    u = np.zeros((order,step))
    mse = np.zeros(step - 1)
    tempx = x
    for i in range(order):
        u[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:20000]

    e[0] = d[0]

    for i in range(1,step):
        K = []
        for j in range(i):
            l = np.linalg.norm(u[:,j] - u[:,i])
            k = exp(-(l**2)/(2*(sigma**2))) 
            K.append(k)
        y[i] = lr*np.dot(e[0:i],K)
        e[i] = d[i] - y[i]
        
        mse[i - 1] = np.sum(e**2)/(i)
    
    mseKLMS = np.mean(e**2)
    return y, mseKLMS

def GKMC(x, d, ss, ks, kp, order):
    step = len(x)

    c = []
    a = []
    e = np.zeros(step)
    y = np.zeros(step)
    u = np.zeros((order, step))
    mse = np.zeros(step)

    tempx = x
    for i in range(order):
        u[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:20000]

    c.append(u[:,0])
    a.append(ss * math.exp(-kp*((d[0])**2)) * d[0])

    for i in range(step):
        s = 0
        for j in range(len(c)):

            l = np.linalg.norm(u[:,j] - u[:,i])
            k = math.exp(-(l**2)/(2*(ks**2))) 
            s = s + a[j]*k
        y[i] = s
        e[i] = d[i] - y[i]

        c.append(u[:,i])
        #a.append(ss * math.exp(-kp*((e[i])**2)) * e[i])
        a.append((ss/(math.sqrt(2*math.pi)* kp**3)) * math.exp(-e[i]**2/(2*(kp**2))) * e[i])
        mse[i] = np.sum(e**2)/(i + 1)
    mseQGKMS = np.mean(e**2)
    return y, mse, mseQGKMS

def GKMCtest(x, d, t, ss, ks, kp, order):
    step = len(x)

    c = []
    a = []
    e = np.zeros(step)
    y = np.zeros(step)
    yt = np.zeros(step)
    ut = np.zeros((order, step))
    u = np.zeros((order, step))
    mse = np.zeros(step)

    tempx = x
    for i in range(order):
        u[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:20000]

    c.append(u[:,0])
    a.append(ss * math.exp(-kp*((d[0])**2)) * d[0])

    for i in range(step):
        s = 0
        for j in range(i):

            l = np.linalg.norm(u[:,j] - u[:,i])
            k = math.exp(-(l**2)/(2*(ks**2))) 
            s = s + a[j]*k
        y[i] = s
        e[i] = d[i] - y[i]

        c.append(u[:,i])
        #a.append(ss * math.exp(-kp*((e[i])**2)) * e[i])
        a.append((ss/(math.sqrt(2*math.pi)* kp**3)) * math.exp(-e[i]**2/(2*(kp**2))) * e[i])
        mse[i] = np.sum(e**2)/(i + 1)
    mseQGKMS = np.mean(e**2)

    tempt = t
    for i in range(order):
        ut[i,:] = tempt
        tempt = np.insert(tempt,0,0)
        tempt = tempt[0:20000]
    
    for i in range(step):
        st = 0
        for j in range(len(c)):
            kt = RBF(c[j], ut[:,i], ks)
            st = st + a[j]*kt
        yt[i] = st
    return yt

def QKLMS(x, d, ss, kw, qs, order):
    step = len(x)

    c = []
    a = []
    e = np.zeros(step)
    y = np.zeros(step)
    u = np.zeros((order, step))
    mse = np.zeros(step)
    #d = d[2:len(d)]
    tempx = x
    for i in range(order):
        u[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:20000]

    c.append(u[:,0])
    a.append(ss*d[0])

    for i in range(step):
        K = []
        s = 0
        for j in range(len(c)):

            k = RBF(c[j], u[:,i], kw)
            K.append(k)
            s = s + a[j]*k
        y[i] = s
        e[i] = d[i] - y[i]

        dis = 100000
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
        mse[i] = np.sum(e**2)/(i + 1)
    mseQKLMS = np.mean(e**2)
    return y, mse, mseQKLMS

def QKLMStest(x, d, t, ss, kw, qs, order):
    step = len(x)

    c = []
    a = []
    e = np.zeros(step)
    y = np.zeros(step)
    ytest = np.zeros(step)
    u = np.zeros((order, step))
    ut = np.zeros((order, step))
    mse = np.zeros(step - 1)
    #d = d[2:len(d)]
    tempx = x
    for i in range(order):
        u[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:20000]

    c.append(u[:,0])
    a.append(ss*d[0])

    for i in range(step):
        K = []
        s = 0
        for j in range(len(c)):
            #c = np.array(c)
            k = RBF(c[j], u[:,i], kw)
            K.append(k)
            s = s + a[j]*k
        y[i] = s
        e[i] = d[i] - y[i]

        dis = 100000
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
            #c = c.tolist()
            c.append(u[:,i])
            a.append(ss*e[i])
        mse[i - 1] = np.sum(e**2)/(i)
    mseQKLMS = np.mean(e**2)

    tempt = t
    for i in range(order):
        ut[i,:] = tempt
        tempt = np.insert(tempt,0,0)
        tempt = tempt[0:20000]
    
    for i in range(step):
        Kt = []
        st = 0
        for j in range(len(c)):
            kt = RBF(c[j], ut[:,i], kw)
            st = st + a[j]*kt
        ytest[i] = st
    
    return ytest

def QGKMC(x, d, ss, kw, qs, cs, kp, order):
    step = len(x)

    c = []
    a = []
    e = np.zeros(step)
    y = np.zeros(step)
    u = np.zeros((order, step))
    mse = np.zeros(step)

    tempx = x
    for i in range(order):
        u[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:20000]

    c.append(u[:,0])
    a.append(ss * math.exp(-kp*(abs(d[0]))**cs) * (abs(d[0]))**(cs - 1) * np.sign(d[0]))

    for i in range(step):
        K = []
        s = 0
        for j in range(len(c)):

            k = RBF(c[j], u[:,i], kw)
            K.append(k)
            s = s + a[j]*k
        y[i] = s
        e[i] = d[i] - y[i]

        dis = 100000
        for j in range(len(c)):
            distemp = np.linalg.norm(u[:,i] - c[j])
            if dis < distemp:
                dis = dis
            else:
                dis = distemp
                jstar = j
        if dis <= qs:
            c = c
            a[jstar] = a[jstar] + ss * math.exp(-kp*(abs(e[i]))**cs) * (abs(e[i]))**(cs - 1) * np.sign(e[i])
        else:

            c.append(u[:,i])
            a.append(ss * math.exp(-kp*(abs(e[i]))**cs) * (abs(e[i]))**(cs - 1) * np.sign(e[i]))
        mse[i] = np.sum(e**2)/(i + 1)
    mseQGKMS = np.mean(e**2)
    return y, mse, mseQGKMS

def QGKMCtest(x, d, t, ss, kw, qs, cs, kp, order):
    step = len(x)

    c = []
    a = []
    e = np.zeros(step)
    y = np.zeros(step)
    yt = np.zeros(step)
    u = np.zeros((order, step))
    ut = np.zeros((order, step))
    mse = np.zeros(step - 1)

    tempx = x
    for i in range(order):
        u[i,:] = tempx
        tempx = np.insert(tempx,0,0)
        tempx = tempx[0:20000]

    c.append(u[:,0])
    a.append(ss * math.exp(-kp*(abs(d[0]))**cs) * (abs(d[0]))**(cs - 1) * np.sign(d[0]))

    for i in range(step):
        K = []
        s = 0
        for j in range(len(c)):

            k = RBF(c[j], u[:,i], kw)
            K.append(k)
            s = s + a[j]*k
        y[i] = s
        e[i] = d[i] - y[i]

        dis = 100000
        for j in range(len(c)):
            distemp = np.linalg.norm(u[:,i] - c[j])
            if dis < distemp:
                dis = dis
            else:
                dis = distemp
                jstar = j
        if dis <= qs:
            c = c
            a[jstar] = a[jstar] + ss * math.exp(-kp*(abs(e[i]))**cs) * (abs(e[i]))**(cs - 1) * np.sign(e[i])
        else:

            c.append(u[:,i])
            a.append(ss * math.exp(-kp*(abs(e[i]))**cs) * (abs(e[i]))**(cs - 1) * np.sign(e[i]))
    tempt = t
    for i in range(order):
        ut[i,:] = tempt
        tempt = np.insert(tempt,0,0)
        tempt = tempt[0:20000]
    
    for i in range(step):
        
        st = 0
        for j in range(len(c)):
            kt = RBF(c[j], ut[:,i], kw)
            st = st + a[j]*kt
        yt[i] = st
    return yt

""" ======================  Variable Declaration ========================== """
lr = 0.2
order = 10
fs = 10000
window = 20000
alpha = 0.1
sigma = 1
xaxis = np.linspace(0, 2, 20000)
xaxisf = np.arange(0,20000)
x2 = np.linspace(0, 2, 19990)
""" ====================== Input Generation ========================== """
x = generate_sinusoid_2(2, 1, 1000, 10000, 0)
d = generate_sinusoid_2(2, 1, 2000, 10000, 0)

""" ============================== i ================================= """
'''yW, mseW = wiener(x, d, order, window)
yLMS, mseLMS = LMS(x, d, lr, order)

fW, tW, zW = signal.stft(yW, fs = 10000)
fLMS, tLMS, zLMS = signal.stft(yLMS, fs = 10000)'''

'''plt.figure(1)
plt.subplot(2,1,1)
plt.plot(xaxis, yW)
plt.title('Time Domin(Wiener Soluiton)')
plt.xlabel('Time(Seconds)')
plt.ylabel('Magnitude')

plt.subplot(2,1,2)
plt.plot(xaxis, yLMS)
plt.title('Time Domin(LMS)')
plt.xlabel('Time(Seconds)')
plt.ylabel('Magnitude')'''


'''plt.figure(2)
plt.subplot(2,1,1)
plt.pcolormesh(tW, fW, abs(zW), vmax = 0.0001)
plt.colorbar()
plt.title('Spectrogram(Wiener Solution)')
plt.xlabel('Time(Seconds)')
plt.ylabel('Frequency(Hz)')

plt.subplot(2,1,2)
plt.pcolormesh(tLMS, fLMS, abs(zLMS), vmax = 0.0001)
plt.colorbar()
plt.title('Spectrogram(LMS)')
plt.xlabel('Time(Seconds)')
plt.ylabel('Frequency(Hz)')'''

'''plt.figure(3)
x1 = ['Wiener Solution', 'LMS']
y1 = [mseW, mseLMS]
plt.bar(x1,y1)
plt.title('MSE between Wiener Solution and LMS')
plt.ylabel("MSE")
plt.show()'''
""" ============================== ii ================================= """
'''yKLMS1, mseKLMS = QKLMS(x, d, 0.1, 1, 1, order)
yKLMS2, mseKLMS = QKLMS(x, d, 0.1, 3, 1, order)
yKLMS3, mseKLMS = QKLMS(x, d, 0.1, 5, 1, order)
yKLMS4, mseKLMS = QKLMS(x, d, 0.1, 10, 1, order)'''

'''stepmse = 15
pltmse = np.zeros(stepmse)
pltx = np.linspace(0.1, 1.5, stepmse)
for i in range(stepmse):
    yKLMS, mseKLMS = QKLMS(x, d, 0.1 + i*0.1, 1, 1, 10)
    pltmse[i] = mseKLMS

plt.figure()
plt.plot(pltx,pltmse)
plt.xlabel("Step Size")
plt.ylabel("MSE")
plt.title("Step Size")
plt.show()'''


'''fKLMS1, tKLMS1, zKLMS1 = signal.stft(yKLMS1, fs = 10000)
fKLMS2, tKLMS2, zKLMS2 = signal.stft(yKLMS2, fs = 10000)
fKLMS3, tKLMS3, zKLMS3 = signal.stft(yKLMS3, fs = 10000)
fKLMS4, tKLMS4, zKLMS4 = signal.stft(yKLMS4, fs = 10000)
plt.figure()
plt.suptitle('the Effect of Step Size')
plt.subplot(2, 2, 1)
plt.pcolormesh(tKLMS1, fKLMS1, abs(zKLMS1), vmax = 0.1)
plt.colorbar()
plt.title('Step Size = 0.0001')
plt.xlabel('Time(Seconds)')
plt.ylabel('Frequency(Hz)')

plt.subplot(2, 2, 2)
plt.pcolormesh(tKLMS2, fKLMS2, abs(zKLMS2), vmax = 0.1)
plt.colorbar()
plt.title('Step Size = 0.001')
plt.xlabel('Time(Seconds)')
plt.ylabel('Frequency(Hz)')

plt.subplot(2, 2, 3)
plt.pcolormesh(tKLMS3, fKLMS3, abs(zKLMS3), vmax = 0.1)
plt.colorbar()
plt.title('Step Size = 0.01')
plt.xlabel('Time(Seconds)')
plt.ylabel('Frequency(Hz)')

plt.subplot(2, 2, 4)
plt.pcolormesh(tKLMS4, fKLMS4, abs(zKLMS4), vmax = 0.1)
plt.colorbar()
plt.title('Step Size = 0.1')
plt.xlabel('Time(Seconds)')
plt.ylabel('Frequency(Hz)')


plt.show()'''

'''ytest, msetest = QKLMS1(x, d, 0.1, 1, 0.1, order)
f, t, z = signal.stft(ytest, fs = 10000)
plt.figure()
plt.pcolormesh(t, f, abs(z))
plt.colorbar()
plt.show()'''

'''t = generate_sinusoid_2(2, 1, 2000, 10000, 0)
ytest1 = QKLMStest(x, d, t, 0.35, 1, 0.001, order)
f0, t0, z0 = signal.stft(t, fs = 10000)
f, t, z = signal.stft(ytest1, fs = 10000)

plt.figure()
plt.suptitle('Test by Input Frequency = 500Hz')
plt.subplot(2,1,1)
plt.pcolormesh(t0, f0, abs(z0))
plt.colorbar()
plt.title('Spectrogram of Input Signal')
plt.xlabel('Time(Seconds)')
plt.ylabel('Frequency(Hz)')

plt.subplot(2,1,2)
plt.pcolormesh(t, f, abs(z))
plt.colorbar()
plt.title('Spectrogram of Output Signal')
plt.xlabel('Time(Seconds)')
plt.ylabel('Frequency(Hz)')
plt.show()'''

""" ============================== iii ================================= """
u1 = np.random.normal(0, 0.1, 20000)
u2 = np.random.normal(4, 0.1, 20000)
u = 0.9*u1 + 0.1*u2
dn = d + u

'''yKLMS1, mseKLMS1 = QKLMS(x, d, 0.1, 1, 1, order)
yKLMS, mseKLMS = QKLMS(x, dn, 0.1, 1, 1, order)

x1 = ['Without Noise', 'With Noise']
y1 = [mseKLMS1, mseKLMS]
plt.bar(x1,y1)
plt.title('the Effect of the Noise')
plt.ylabel("MSE")
plt.show()'''

'''yKLMS, mseKLMS, MSEKLMS = QKLMS(x, dn, 0.2, 2, 0.01, order)
yMCLMS, mseMCLMS, MSEQGKMC = GKMC(x, dn, 0.2, 2, 1, order)


plt.figure()
plt.plot(xaxisf, mseKLMS, label = 'MSE-KLMS')
plt.plot(xaxisf, mseMCLMS, label = 'MCC-KLMS')
plt.title('MSE between MSE-KLMS and MCC-KLMS')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.legend()
plt.show()'''


'''stepmse = 10
pltmse = np.zeros(stepmse)
pltx = np.linspace(0.01, 0.1, stepmse)
for i in range(stepmse):
    yKLMS, mseKLMS = QGKMC(x, dn, 0.2, 1.25, 0.01, 1, 1.6, 10)
    pltmse[i] = mseKLMS

plt.figure()
plt.plot(pltx,pltmse)
plt.xlabel("Kernel Parameter Lambda")
plt.ylabel("MSE")
plt.title("Kernel Parameter Lambda")
plt.show()'''


#compare and draw
'''yKLMS, mseKLMS = QKLMS(x, dn, 0.2, 3, 1, order)
yMCLMS, mseMCLMS = QGKMC(x, dn, 0.2, 3, 1, 0.5, 1, order)
fLMS, tLMS, zLMS = signal.stft(yKLMS, fs = 10000)
fMCLMS, tMCLMS, zMCLMS = signal.stft(yMCLMS, fs = 10000)

plt.figure()

plt.subplot(2, 1, 1)
plt.pcolormesh(tLMS, fLMS, abs(zLMS))
plt.colorbar()
plt.subplot(2, 1, 2)
plt.pcolormesh(tMCLMS, fMCLMS, abs(zMCLMS))
plt.colorbar()
plt.show()'''


#test generalization
t = generate_sinusoid_2(2, 1, 2000, 10000, 0)

ytest1 = QKLMStest(x, dn, t, 0.2, 2, 0.1, order)
#ytest2 = QGKMCtest(x, dn, t, 0.2, 1.25, 0.001, 2, 1.6, order)
ytest2 = GKMCtest(x, dn, t, 0.2, 2, 1.5, order)
f1, t1, z1 = signal.stft(ytest1, fs = 10000)
f2, t2, z2 = signal.stft(ytest2, fs = 10000)

'''plt.figure()
plt.plot(xaxisf, ytest2)
plt.plot(xaxisf, x)
plt.show()'''


plt.figure()
plt.subplot(2,1,1)
plt.pcolormesh(t1, f1, abs(z1), vmax = 0.1)
plt.colorbar()
plt.title('Spectrogram of MSE-KLMS')
plt.xlabel('Time(Seconds)')
plt.ylabel('Frequency(Hz)')

plt.subplot(2,1,2)
plt.pcolormesh(t2, f2, abs(z2), vmax = 0.1)
plt.colorbar()
plt.title('Spectrogram of MCC-KLMS')
plt.xlabel('Time(Seconds)')
plt.ylabel('Frequency(Hz)')
plt.show()
a=1