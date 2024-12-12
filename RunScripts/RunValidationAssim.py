import sys
sys.path.append('src')
sys.path.append('../src')
import numpy as np
from src.EnKF import *
from src.L96 import *
import pickle
from src.Experiment import *
from src.NeuralNet import *
from src.NeuralNet import *
import random
import tensorflow as tf
import yaml

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

def runaug(sfile):
    with open(sfile) as f:
        settings = yaml.safe_load(f)
    fn = settings['experiment_path']
    name = settings['name']

    with open(fn,'rb') as f:
        base = pickle.load(f)
    with open(fn, 'rb') as f:
        experiment = pickle.load(f)
    npoints = base.ds.xx.shape[1]-1
    npoints = int(npoints)
    x0 = base.ds.xaens[:,:,int(npoints/2)]
    del(base)

    params = settings['experiment_settings']
    freq = params['freq']
    experiment.frac = params['frac']
    experiment.assim_method = params['assim_method']
    experiment.alpha = params['alpha']
    experiment.ds['yy'] = experiment.ds['yy'][:, int(npoints / 2)::freq]
    experiment.dt = experiment.dt*freq

    experiment.loc_method = 'original'

    try:
        nn_fn = settings['nn_fname']
        with open(nn_fn) as f:
            nn_settings = yaml.safe_load(f)
        nlayers = nn_settings['nlayers']
        filter_size = nn_settings['filter_size']
        nfmaps = nn_settings['nfmaps']
        nn = NeuralNet(nlayers=nlayers,filter_size=filter_size,nfmaps=nfmaps,N=40)
        nn.buildmodel()
        nn.model.load_weights(settings['weights_file'])
    except KeyError:
        nn = None
    xaens,xfens = experiment.assimilate(x0, nn)
    rmse_a = rms(experiment.ds.xx[:,int(npoints/2)::freq],xaens.mean(axis=1))
    print('Analysis RMSE for run '+name+': '+str(rmse_a.mean()))

    if experiment.nens>100:
        with open(name+'.pickle','wb') as f:
            pickle.dump(experiment,f,protocol=4)
    else:
        with open(+name+'.pickle','wb') as f:
            pickle.dump(experiment,f)

fname = "./validationrunsettings.yml"
runaug(fname)