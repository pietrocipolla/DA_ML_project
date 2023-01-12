import numpy as np
from keras import Sequential
from keras.layers import *
import keras
from DA import *
from matplotlib import pyplot as plt

'Script for training CNN on EnKF analysis data'
np.random.seed(0)

def go(name,experiment):
    with open('Sensitivity/experiment','rb') as f:
        exp = pickle.load(f)
    xa = exp.xa.mean(axis=2).transpose()
    xf = np.loadtxt('Sensitivity6/xf_base.txt')
    xx = np.loadtxt('Sensitivity6/xx_base.txt')
    yy = np.loadtxt('Sensitivity6/yy_base.txt')
    #var = np.loadtxt('Sensitivity/var.txt')
    nvars = xf.shape[1]

    model = Sequential()
    model.add(Conv1D(5,3,activation='softmax',input_shape=(nvars+6,2)))
    model.add(Conv1D(5,3,activation='softmax'))
    model.add(Conv1D(1,3,activation=None))

    nsamples = xx.shape[0]
    nepochs  = 10
    ntest = nsamples*.5
    ntest = int(ntest)
    ntrain = nsamples-ntest
    X = np.zeros([nsamples,nvars+6,2])

    X[:,3:(nvars+3),0] = (xf-xx.min())/xx.max()
    X[:,:3,0] = (xf[:,(nvars-3):]-xx.min())/xx.max()
    X[:,(nvars+3):,0] = (xf[:,:3]-xx.min())/xx.max()

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
    model.save_weights('TrainedModels/'+name)
    return(res,xtest,ytest)

res = go(name='normtest')