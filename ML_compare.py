import numpy as np

from DA import *
from matplotlib import pyplot as plt
import random
import time
from keras import Sequential
from keras.layers import *
from Model import Model

ml_metrics = {}
enkf_metrics = {}
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



m = Model(N, F)

#load machine learning model
ml = Sequential()
ml.add(Conv1D(5,3,activation='softplus',input_shape=(16,2)))
ml.add(Conv1D(5,3,activation='softplus'))
ml.add(Conv1D(1,3,activation=None))
ml.load_weights('TrainedModels/m_base')

x_f = np.loadtxt('TrainingData/xf1.txt')
x_a = np.loadtxt('TrainingData/xa1.txt')
xx = np.loadtxt('TrainingData/xx1.txt')
yy = np.loadtxt('TrainingData/yy1.txt')

n_timesteps = len(t)
n_test = int(n_timesteps/2)
x0 = x_a[n_test-1,:] #initial x is last analysis of training dataset
yy_test = yy[n_test:,:]
xa_test = np.zeros(yy_test.shape)
xf_test = np.zeros(yy_test.shape)
xa_test = xa_test.transpose()

#variables not used:
h = 0
sigma_z = 0
gamma = 0
nz = yy.shape[0]


daml = DA(m.M,h,sigma_z,sigma_x,gamma)
#"ensemble size" of 1 for ml application:
daml.initialize(x0, loc=0, scale=0, n_ens=1)
daml.x0[:,0] = x0

for i in range(n_test,nz):
    t0 = t[i - 1]
    t1 = t[i]
    daml.forecast(t0, t1)
    #frac = 1
    #hi = random.choices(idx, k=int(N * frac))
    z = yy[i, :]

    xa = daml.assim_ann(ml,daml.xf, z)
    xa = xa[0,:,0]
    xf_test[i-n_test, :] = daml.xf[:,0]
    xa_test[:, i-n_test] = xa
    daml.x0 = xa.reshape(10,1)
    if i % 100 == 0:
        print(t[i])

xa_test = xa_test.transpose()

errors_nn = xa_test-xx[n_test:,:]
std_nn = errors_nn.std()
rms_nn = errors_nn**2
rms_nn = rms_nn.sum(axis=1)
rms_nn = rms_nn/10
rms_nn = np.sqrt(rms_nn)

print(rms_nn.mean())
print(errors_nn.std())

