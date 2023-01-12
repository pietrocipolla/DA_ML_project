import numpy as np
from keras import Sequential
from keras.layers import *
import keras
import pickle
from matplotlib import pyplot as plt

'Script for training CNN on EnKF analysis data'
np.random.seed(0)

def go(name,experiment):
    with open(experiment,'rb') as f:
        exp = pickle.load(f)
    xa = exp.xa.mean(axis=2)
    xf = exp.xf.mean(axis=2)
    xx = exp.xx[:,1:].transpose()
    yy = exp.yy[:,1:].transpose()
    #var = np.loadtxt('Sensitivity/var.txt')
    nvars = xf.shape[1]
    model = Sequential()
    model.add(Conv1D(5,3,activation='relu',input_shape=(nvars+6,2)))
    model.add(Conv1D(5,3,activation='relu'))
    model.add(Conv1D(1,3,activation=None))

    nsamples = xx.shape[0]
    nepochs  = 10
    ntest = nsamples*.5
    ntest = int(ntest)
    ntrain = nsamples-ntest
    X = np.zeros([nsamples,nvars+6,2])
    X[:,3:(nvars+3),0] = xf
    X[:,:3,0] = (xf[:,(nvars-3):])
    X[:,(nvars+3):,0] = (xf[:,:3])

    X[:,3:(nvars+3),1] = yy[:,:] - xf
    X[:,:3,1] = yy[:,(nvars-3):] -xf[:,(nvars-3):]
    X[:,(nvars+3):,1] = yy[:,:3] - xf[:,:3]
    #X[:,:,1] = X[:,:,1] - X[:,:,0]

    yy = yy.reshape(nsamples,nvars,1)
    xx1 = xx.reshape(nsamples,nvars,1)
    xa = xa.reshape(nsamples,nvars,1)
    ytrain = xa[:ntrain,:,:]
    xtrain = X[:ntrain,:,:]
    ytest = xx1[ntrain:,:,:]
    xtest = X[ntrain:,:,:]
    model.compile(optimizer='SGD',loss='mean_squared_error',metrics=[keras.metrics.RootMeanSquaredError()])
    res = model.fit(xtrain,ytrain,epochs=nepochs,validation_data=[xtest,ytest])
    model.save_weights(name)
    return(res,xtest,ytest)

res = go(name='test',experiment=r'Sensitivity/s9.pickle')