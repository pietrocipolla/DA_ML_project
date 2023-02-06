import numpy as np
from Experiment import *
import xarray as xr
from tqdm import tqdm
from L96 import *
import pickle
import tensorflow as tf
import random

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)
deltat = 0.05

with open('Data/base.pickle','rb') as f:
    base = pickle.load(f)
with open('Data/a_base.pickle','rb') as f:
    hybrid = pickle.load(f)
with open('Data/sparse_base.pickle','rb') as f:
    sparse = pickle.load(f)
N = int(len(base.ds.space))



def forecast(ensemble,truth,interval,max_t):
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

nsamples = 1000
ntimesteps = 40000
idx = list(range(int(ntimesteps/2),ntimesteps+1,2))
np.random.shuffle(idx)
idx = idx[:nsamples]

xaens = hybrid.ds.xaens.data
xx = hybrid.ds.xx.data

ret_nn = forecast(xaens[:,:,idx],xx[:,idx],deltat,3)

xaens = sparse.ds.xaens.data
ret_sparse = forecast(xaens[:,:,idx],xx[:,idx],deltat,3)


xaens = base.ds.xaens.data
ret_dense = forecast(xaens[:,:,idx],xx[:,idx],deltat,3)

np.save('Data/sparse_errors_1000.npy',ret_sparse)
np.save('Data/nn_errors_1000.npy',ret_nn)
np.save('Data/dense_errors.npy',ret_dense)