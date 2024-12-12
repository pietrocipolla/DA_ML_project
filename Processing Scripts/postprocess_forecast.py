import sys
sys.path.append('src')
sys.path.append('../src')
import numpy as np
from Experiment import *
from tqdm import tqdm
from L96 import *
import pickle
import tensorflow as tf
import random

'''
Script for computing forecast accuracy from initial conditions from AllObs, Augmented, and EnKF SparseObs
'''

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)
deltat = 0.05

#Allobs results:
with open('RawResults/Sensitivity/base.pickle','rb') as f:
    base = pickle.load(f)
#Augmented results
with open('RawResults/Validation/a_base.pickle','rb') as f:
    augmented = pickle.load(f)
#SparseObs Results
with open('RawResults/Validation/sparse_base.pickle','rb') as f:
    sparse = pickle.load(f)
N = 40



def forecast(ensemble,truth,interval,max_t):
    '''
    Function for computing ensemble forecasts from ensemble initial conditions.

    Args
    ensemble: numpy array, initial condition ensembles
    truth: true initial conditions
    interval: time step for forecast output
    max_t: maximum forecast lead time in model units
    '''
    t = np.arange(0.0, max_t,interval)
    n_ens = ensemble.shape[1]
    N = ensemble.shape[0]
    n_t0 = ensemble.shape[2]
    m = L96(F=8,N=N)
    errors = np.zeros([n_t0,len(t),n_ens,N])
    for i in tqdm(range(n_t0)):
        x_true = m.integrate(truth[:,i],t)
        for j in range(n_ens):
            ret = m.integrate(ensemble[:,j,i],t)
            errors[i,:,j,:] = (ret-x_true).transpose()
    return(errors)

#pick 1000 random time steps to use as initial conditions
nsamples = 1000
ntimesteps = 40000
idx = list(range(int(ntimesteps/2),ntimesteps+1,2))
np.random.shuffle(idx)
idx = idx[:nsamples]

#Retrieve augmented ensemble analyses
xaens = augmented.ds.xaens.data
xx = augmented.ds.xx.data #truth (only needs to be loaded once, same for all)

#compute forecasts
ret_aug = forecast(xaens[:,:,idx],xx[:,idx],deltat,3)

#sparse analyses:
xaens = sparse.ds.xaens.data
ret_sparse = forecast(xaens[:,:,idx],xx[:,idx],deltat,3)

#allobs analyses
xaens = base.ds.xaens.data
ret_dense = forecast(xaens[:,:,idx],xx[:,idx],deltat,3)

#save results
np.save('sparse_errors.npy',ret_sparse)
np.save('aug_errors.npy',ret_aug)
np.save('dense_errors.npy',ret_dense)