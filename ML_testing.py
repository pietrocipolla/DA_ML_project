import numpy as np
from matplotlib import pyplot as plt
import time
from keras import Sequential
from keras.layers import *
import keras
from keras import metrics
from DA import *
from Model import Model


def go(name):
    ml = Sequential()
    ml.add(Conv1D(5,3,activation='softplus',input_shape=(16,2)))
    ml.add(Conv1D(5,3,activation='relu'))
    ml.add(Conv1D(1,3,activation=None))
    ml.load_weights('TrainedModels/'+name)

    np.random.seed(0)
    runparams = {'N':10,'F':8,'tmax':4000,'deltat':0.1,'ens_size':1,'sigma_x':0.1,'r':1,'gamma':1,'loc':4}
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
    xx = np.loadtxt('TrainingData/xx1.txt')
    yy = np.loadtxt('TrainingData/yy1.txt')


    ## generate array to hold analysis and forecast
    idx = range(N)
    x_a = np.zeros(xx.shape)
    x_f = np.zeros(xx.shape)
    x_a = x_a.transpose()
    enkf = DA(m.M, 0, s, sigma_x=sigma_x, gamma=gamma, loc=loc)
    enkf.initialize(x0, loc=0, scale=sigma_x, n_ens=ens_size)

    x_a[:, 0] = enkf.x0[:,0]
    nz = yy.shape[0]
    en_std = []
    for i in range(1, nz):
        t0 = t[i - 1]
        t1 = t[i]
        enkf.forecast(t0, t1)
        #frac = 1
        #hi = random.choices(idx, k=int(N * frac))
        z = yy[i, :]

        xa = enkf.assim_ann(ml,enkf.xf, z)
        xa = xa[0,:,0]
        x_f[i, :] = enkf.xf[:,0]
        x_a[:, i] = xa
        enkf.x0 = xa.reshape(10,1)
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
    return(rms(rmsa).mean()/rms(rmsz).mean())

def rms(x):
    ret = x**2
    n = x.shape[1]
    ret = ret.sum(axis=1)
    ret = ret/n
    ret = np.sqrt(ret)
    return(ret)

ret = go(name='mm')