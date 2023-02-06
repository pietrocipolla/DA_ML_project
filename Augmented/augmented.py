import numpy as np
from matplotlib import pyplot as plt
from EnKF import *
from L96 import *
import time
import pickle
from Experiment import *
from NeuralNet import *
import keras
import xarray as xr
from NeuralNet import *
import random
import tensorflow as tf

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
def rms(xx,xa):
    ss = xx-xa
    ss = ss**2
    ss = ss.sum(axis=0)
    ss = ss/40
    ss = np.sqrt(ss)
    return(ss)

def runaug(case):
    if case == 0:
        name = 'base'
        fn = r'/Users/luho4863/PycharmProjects/ML_DA_TDD/Sensitivity/base.pickle'
    else:
        fn = r'/Users/luho4863/PycharmProjects/ML_DA_TDD/Sensitivity/s'+str(case)+'.pickle'
        name = str(case)

    with open(fn,'rb') as f:
        base = pickle.load(f)
    with open(fn, 'rb') as f:
        experiment = pickle.load(f)
    npoints = base.ds.xx.shape[1]-1
    npoints = int(npoints)
    x0 = base.ds.xaens[:,:,int(npoints/2)]
    experiment.frac = 0.25
    experiment.ds['yy'] = experiment.ds['yy'][:,int(npoints/2):]
    nn = NeuralNet(nlayers=3,filter_size=3,N=40)
    nn.buildmodel()
    nn.model.load_weights('m9.hd5')
    xaens,xfens = experiment.assimilate(x0, nn)
    rmse_a = rms(experiment.ds.xx[:,int(npoints/2):],xaens.mean(axis=1))
    with open('a_'+name+'.pickle','wb') as f:
        pickle.dump(experiment,f)
    print('augmented rmse:')
    print(float(rmse_a.mean()))

    with open(fn, 'rb') as f:
        experiment = pickle.load(f)
    experiment.ds['yy'] = experiment.ds['yy'][:,int(npoints/2)::2]
    experiment.frac = 0.25
    experiment.dt = 0.1
    xaens,xfens = experiment.assimilate(x0)
    rmse_s = rms(experiment.ds.xx[:,int(npoints/2)::2],xaens.mean(axis=1))
    with open('sparse_'+name+'.pickle','wb') as f:
        pickle.dump(experiment,f)
    print('sparseobs rmse:')
    print(float(rmse_s.mean()))

for case in range(10):
    runaug(case)