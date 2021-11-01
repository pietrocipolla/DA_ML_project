import numpy as np
from keras import Sequential
from keras.layers import *
import keras
from DA import *
from matplotlib import pyplot as plt

'Script for training CNN on EnKF analysis data'
np.random.seed(0)

def go(name):
    model = Sequential()
    model.add(Conv1D(5,3,activation='softplus',input_shape=(16,2)))
    model.add(Conv1D(5,3,activation='softplus'))
    model.add(Conv1D(1,3,activation=None))

    xa = np.loadtxt('TrainingData/xa1.txt')
    xf = np.loadtxt('TrainingData/xf1.txt')
    xx = np.loadtxt('TrainingData/xx1.txt')
    yy = np.loadtxt('TrainingData/yy1.txt')

    nsamples = xx.shape[0]
    nepochs  = 10
    ntest = nsamples*.5
    ntest = int(ntest)
    ntrain = nsamples-ntest
    X = np.zeros([nsamples,16,2])
    X[:,3:13,0] = xf
    X[:,:3,0] = xf[:,7:]
    X[:,13:,0] = xf[:,:3]

    X[:,3:13,1] = yy[:,:] - xf
    X[:,:3,1] = yy[:,7:] -xf[:,7:]
    X[:,13:,1] = yy[:,:3] - xf[:,:3]
    #X[:,:,1] = X[:,:,1] - X[:,:,0]

    yy = yy.reshape(nsamples,10,1)
    xa = xa.reshape(nsamples,10,1)
    ytrain = xa[:ntrain,:,:]
    xtrain = X[:ntrain,:,:]
    ytest = xa[ntrain:,:,:]
    xtest = X[ntrain:,:,:]
    model.compile(optimizer='SGD',loss='mean_squared_error',metrics=[keras.metrics.RootMeanSquaredError()])
    res = model.fit(xtrain,ytrain,epochs=10)
    model.save_weights('TrainedModels/'+name)

go(name='m_base')

'''
plt.subplots(1,2)
plt.gcf().set_size_inches(10,5)
plt.subplot(1,2,1)
plt.plot(np.array(range(1,nepochs+1)),res.history['root_mean_squared_error'],'k',label='Training')
plt.plot(np.array(range(1,nepochs+1)),res.history['val_root_mean_squared_error'],'r',label='Testing')
plt.legend()
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('RMS Error',fontsize=12)

plt.subplot(1,2,2)
plt.plot(np.array(range(1,nepochs+1)),res.history['loss'],'k',label='Training')
plt.plot(np.array(range(1,nepochs+1)),res.history['val_loss'],'r',label='Testing')
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('Sum. Sqd. Error',fontsize=12)
plt.legend()

plt.show()

'''