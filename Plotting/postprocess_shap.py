import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from keras import Sequential
from keras.layers import *
import shap
import pickle

N=40
ml = Sequential()
ml.add(Conv1D(5,3,activation='relu',input_shape=(N+6,2)))
ml.add(Conv1D(5,3,activation='relu'))
ml.add(Conv1D(1,3,activation=None))
ml.load_weights('Data/m9.hd5')

with open(r'/Users/luho4863/PycharmProjects/ML_DA_TDD/Sensitivity/base.pickle','rb') as f:
    base = pickle.load(f)
xa = base.ds.xaens.mean(axis=1).data.transpose()[1:]
xf = base.ds.xfens.mean(axis=1).data.transpose()[1:]
xx = base.ds.xx.data.transpose()[1:]
yy = base.ds.yy.data.transpose()[1:]
tmax = 2000
deltat = 0.05
t = np.arange(0.0, tmax, deltat)


nsamples = xx.shape[0]
ntest = nsamples*.5
ntest = int(ntest)
ntrain = nsamples-ntest
X = np.zeros([nsamples,N+6,2])
X[:,3:(N+3),0] = xf
X[:,:3,0] = xf[:,(N-3):]
X[:,(N+3):,0] = xf[:,:3]

X[:,3:(N+3),1] = yy[:,:] - xf
X[:,:3,1] = yy[:,(N-3):] -xf[:,(N-3):]
X[:,(N+3):,1] = yy[:,:3] - xf[:,:3]
ytrain = xa.reshape(nsamples,N,1)[:ntrain,:,:]
xtrain = X[:ntrain,:,:]
ytest = xa.reshape(nsamples,N,1)[ntrain:,:,:]
xtest = X[ntrain:,:,:]
xin = np.zeros([int(nsamples/2),2*(N+6)])
xin[:,:N+6] = xtrain[:,:,0]
xin[:,N+6:] = xtrain[:,:,1]
xout = np.zeros([int(nsamples/2),2*(N+6)])
xout[:,:(N+6)] = xtest[:,:,0]
xout[:,(N+6):] = xtest[:,:,1]

def f(x):
    dim = x.shape
    xout = np.zeros([dim[0],int(dim[1]/2),2])
    xout[:,:,0] = x[:,:N+6]
    xout[:,:,1] = x[:,N+6:]
    out = ml.predict(xout)
    return(out)


explainer=shap.KernelExplainer(f,data=xin.mean(axis=0).reshape([1,2*(N+6)]))
ret = explainer.shap_values(xout[:1000])
ret = np.array(ret)
ret = ret[:,:,list(range(3,N+3))+list(range(N+9,2*N+9))]
nsamples = ret[0].shape[0]
ndist = 4
ret2 = np.zeros([nsamples,2*(ndist+1)])
for i in range(N):
    ret2[:,0] += ret[i,:,i]/N
    for j in range(1,ndist+1):
        dist = j
        idx = i+j
        if idx>(N-1):
            idx = idx-N
        ret2[:,j] += ret[i,:,idx]/(2*N)
        idx = i-dist
        if idx<0:
            idx=idx+N
        ret2[:,j] +=ret[i,:,idx]/(2*N)

    ret2[:,ndist+1] += ret[i,:,i+N]/N
    for j in range(1,ndist+1):
        dist = j
        idx = i+j+N
        if idx>(2*N-1):
            idx = idx-N
        ret2[:,ndist+1+j] += ret[i,:,idx]/(2*N)
        idx = i-dist
        if idx<N:
            idx=idx+N
        ret2[:,j+ndist+1] +=ret[i,:,idx]/(2*N)

names=['$x_{0}$','$x_{i-1}$','$x_{i-2}$','$x_{i-3}$','$x_{i-4}$',r'$\delta x_{0}$',r'$\delta x_{i-1}$',r'$\delta x_{i-2}$',r'$\delta x_{i-3}$',r'$\delta x_{i-4}$']
np.savetxt('Data/shapvals.txt',ret2)