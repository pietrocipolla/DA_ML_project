import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from keras import Sequential
from keras.layers import *
import shap
import pickle
from NeuralNet import *

'''
Script for calculating SHAP values from NN model.

The file is set up to compute SHAP values for the trained NN presented in the published paper.
'''

# Build trained NN, and load weights from training iteration 24
N=40
nn = NeuralNet(5,3,40,7)
nn.buildmodel()
ml = nn.model
ml.load_weights('progress/iter_47weights.23.hdf5')

# Load base analysis, forecast, ensemble standard deviation, truth, and obs.
with open('../Sensitivity/base.pickle', 'rb') as f:
    base = pickle.load(f)
xa = base.ds.xaens.mean(axis=1).data.transpose()[1:]
xf = base.ds.xfens.mean(axis=1).data.transpose()[1:]
xastd = base.ds.xaens.data.std(axis=1).transpose()[1:, :]
xstd = base.ds.xfens.data.std(axis=1).transpose()[1:, :]
xx = base.ds.xx.data.transpose()[1:]
yy = base.ds.yy.data.transpose()[1:]
tmax = 2000
deltat = 0.05
t = np.arange(0.0, tmax, deltat)


#Build input pipeline
nsamples = xx.shape[0]
ntest = nsamples*.5
ntest = int(ntest)
ntrain = nsamples-ntest
xin = np.zeros([nsamples, N, 2])
xin[:, :, 0] = xf
xin[:, :, 1] = xstd
yout = np.zeros([nsamples,N, 2])
yout[:, :, 0] = xa
yout[:, :, 1] = xastd
X = nn.make_input(xin,yy)
targets = nn.maketargets(yout)
ytrain = targets[:ntrain]
xtrain = X[:ntrain,:,:]
ytest = targets[:ntrain]
xtest = X[ntrain:,:,:]
xin = np.zeros([int(nsamples/2),3*N])
xin[:,:N] = xf[:ntrain]
xin[:,N:2*N] = xstd[:ntrain]
xin[:,2*N:] = yy[:ntrain]
xout = np.zeros([int(nsamples/2),3*N])
xout[:,:N] = xf[ntrain:]
xout[:,N:2*N] = xstd[ntrain:]
xout[:,2*N:] = yy[ntrain:]

#Define function that maps input vectors to output vectors for SHAP analysis
def f(x):
    dim = x.shape
    if len(dim)>1:
        nsamples = dim[0]
        xi = np.zeros([nsamples,N,3])
        xi[:,:,0] = x[:,:N]
        xi[:,:,1] = x[:,N:2*N]
        yi = x[:,2*N:]
        yi = yi.reshape([nsamples,N])
        out = ml.predict(nn.make_input(xi,yi))
        ret = np.zeros([nsamples,2*N])
        ret[:,:N] = out[:,:,0]
        ret[:,N:] = out[:,:,1]
    return(ret)

#Compute SHAP values for first 1000 validation data points, and save results to .npy file.
explainer=shap.KernelExplainer(f,data=xin.mean(axis=0).reshape([1,3*(N)]))

#ret is a np array of shape [80,1000,120], with the first axis being output variables, the second samples, and the third input variables.
ret = explainer.shap_values(xout[:1000])
ret = np.array(ret)
np.save('shapvals.npy',ret)

