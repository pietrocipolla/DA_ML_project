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
import tensorflow as tf
import random
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

nn = NeuralNet(nlayers=3,filter_size=3,N=40)
with open('TrainingData/s9.pickle','rb') as f:
    s9 = pickle.load(f)
nn.buildmodel()
nn.train(training_fraction=0.5, nepochs=20, optimizer='sgd', experiment=s9.ds)
fn = 'cnn_weights.hd5'
nn.model.save_weights(fn)
val_rms = nn.model.history.history['val_root_mean_squared_error']
train_rms = nn.model.history.history['root_mean_squared_error']
np.savetxt('val_rms.txt',np.array(val_rms))
np.savetxt('train_rms.txt',np.array(train_rms))