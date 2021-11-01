import numpy as np
from DA import *
from Model import Model
from matplotlib import pyplot as plt
import random
import time

'''
Script for running EnKF on Lorenz96 model. Run settings initialized at top of script.

Localization and covariance inflation are included
'''

np.random.seed(0)
runparams = {'N':10,'F':8,'tmax':2000,'deltat':0.3,'ens_size':20,'sigma_x':0.1,'r':.1,'gamma':1,'loc':3}
N = runparams['N']  # number of variables
F = runparams['F']  # forcing parameter
x0 = 8 * np.ones(N)  # initial state
x0[0] += 0.01
tmax = runparams['tmax']  # maximum time of simulation
deltat = runparams['deltat']  # output interval
t = np.arange(0.0, tmax, deltat)
m = Model(N, F)
ens_size = runparams['ens_size']  # ensemble size
sigma_x = runparams['sigma_x']  # model standard deviation
r = runparams['r']  # measurement variance
s = np.sqrt(r)
gamma = runparams['gamma']  # inflation parameter
loc = runparams['loc']
frac = 1


a = time.time()
# generate true values
xx = m.M(x0, t)
#xx = np.loadtxt('data/xx.txt')

# generate measurements
yy = xx + np.random.normal(loc=0, scale=s, size=xx.shape)

## generate array to hold analysis and forecast
idx = range(N)
x_a = np.zeros(xx.shape)
x_f = np.zeros(xx.shape)
x_a = x_a.transpose()
enkf = DA(m.M, 0, s, sigma_x=sigma_x, gamma=gamma, loc=loc)
enkf.initialize(x0, loc=0, scale=sigma_x, n_ens=ens_size)

x_a[:, 0] = enkf.x0.mean(axis=1)
nz = yy.shape[0]
en_std = []
for i in range(1, nz):
    t0 = t[i - 1]
    t1 = t[i]
    enkf.forecast(t0, t1)
    hi = random.choices(idx, k=int(N * frac))
    hi = list(range(10))
    z = yy[i, hi]
    k = enkf.getK(hi)
    xa = enkf.assim(enkf.xf, hi, z, k)
    x_f[i, :] = enkf.xf.mean(axis=1)
    x_a[:, i] = xa.mean(axis=1)
    enkf.x0 = xa
    en_std.append(xa.std(axis=1).mean())
    if i % 100 == 0:
        print(t[i])
        '''
        plt.close('all')
        rmsa = x.transpose() - xx
        rmsz = yy - xx
        rmsz = rmsz[:i,:]
        rmsa = rmsa[:i,:]
        plt.subplots(1, 2)
        plt.subplot(1, 2, 1)
        plt.plot(t[:i],rmsz.std(axis=1), 'k-')
        plt.plot(t[:i],rmsa.std(axis=1), 'r-')
        plt.xlim(0,tmax)
        plt.subplot(1, 2, 2)
        plt.plot(t[:i],en_std)
        plt.xlim(0,tmax)
        plt.show(block=False)
        '''
    else:
        pass

rmsa = x_a.transpose() - xx
rmsz = yy - xx
rmsf = x_f-xx

def rms(x):
    ret = x**2
    n = x.shape[1]
    ret = ret.sum(axis=1)
    ret = ret/n
    ret = np.sqrt(ret)
    return(ret)
print('time:')
print(time.time()-a)

print(rms(rmsa).mean()/rms(rmsf).mean())
print(rms(rmsa).mean()/rms(rmsz).mean())

plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.plot(t[1:], rms(rmsa)[1:], 'k-')
plt.plot(t[1:], rms(rmsf)[1:], 'r-')
plt.plot(t[1:], rms(rmsz)[1:], 'g-')
plt.xlim(0, tmax)
plt.subplot(1, 2, 2)
plt.plot(t[1:], en_std)
plt.xlim(0, tmax)
plt.show()

#np.savetxt('TrainingData/xx1.txt',xx)
#np.savetxt('TrainingData/yy1.txt',yy)
#np.savetxt('TrainingData/xa1.txt',x_a.transpose())
#np.savetxt('TrainingData/xf1.txt',x_f)
with open('TrainingData/settings.txt','w') as f:
    f.write(str(runparams))
