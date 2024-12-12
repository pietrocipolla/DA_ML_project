import sys
sys.path.append('src')
sys.path.append('../src')
import numpy as np
from src.EnKF import *
from src.L96 import *
import pickle
from src.Experiment import *
from src.NeuralNet import *
import tensorflow as tf
import random
import yaml
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

def rms(xx,xa):
    ss = xx-xa
    ss = ss**2
    ss = ss.sum(axis=1)
    ss = ss/40
    ss = np.sqrt(ss)
    return(ss)

if __name__ == '__main__':
    settings_fname = '../NNTuning/m_47.yml'
    with open(settings_fname) as f:
        settings = yaml.safe_load(f)
    nlayers = settings['nlayers']
    filter_size = settings['filter_size']
    nfmaps = settings['nfmaps']
    nvars = settings['nvars']
    optimizer = settings['optimizer']
    training_fraction = settings['training_fraction']
    nepochs = settings['nepochs']
    nn = NeuralNet(nlayers=nlayers,filter_size=filter_size,N=nvars,nfmaps=nfmaps)
    with open('./s9.pickle','rb') as f:
        s9 = pickle.load(f)
    nn.buildmodel()
    nn.train(training_fraction=training_fraction, nepochs=nepochs, optimizer=optimizer, experiment=s9.ds)

    # experiment = s9.ds
    # del(s9)
    # xa = experiment.xaens.data.mean(axis=1).transpose()[1:, :]
    # xastd = experiment.xaens.data.std(axis=1).transpose()[1:, :]
    # xf = experiment.xfens.data.mean(axis=1).transpose()[1:, :]
    # xstd = experiment.xfens.data.std(axis=1).transpose()[1:, :]
    # xx = experiment.xx.data[:, 1:].transpose()
    # yy = experiment.yy.data[:, 1:].transpose()
    # nsamples = xx.shape[0]
    # nvars = xx.shape[1]
    # training_fraction = 0.5
    # ntrain = int(nsamples * training_fraction)
    # xin = np.zeros([nsamples, nvars, 2])
    # xin[:, :, 0] = xf
    # xin[:, :, 1] = xstd
    # yout = np.zeros([nsamples, nvars, 2])
    # yout[:, :, 0] = xa
    # yout[:, :, 1] = xastd
    # xtest = nn.make_input(xin[ntrain:], yy[ntrain:, :])
    # ytest = nn.maketargets(yout[ntrain:])
    # ypred = nn.model.predict(xtest)
    # pred_er = (xx[ntrain:,:]-ypred[:,:,0]).reshape(800000)
    # enkf_er = (xx[ntrain:,:]-xa[ntrain:,:]).reshape(800000)
    try:
        name = settings['name']
    except KeyError:
        name = 'model'
    fn = name+'.weight.h5'
    settings['weights_file'] = fn
    nn.model.save_weights(fn)

    val_rms = nn.model.history.history['val_root_mean_squared_error'][-1]
    settings['val_rms'] = val_rms
    with open('../NNTuning/'+name+'_results.yml','w') as f:
        yaml.dump(settings,f)
    train_rms = nn.model.history.history['root_mean_squared_error']