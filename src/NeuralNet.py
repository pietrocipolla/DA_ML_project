import keras
import numpy as np
from keras import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint


class NeuralNet:
    def __init__(self, nlayers=None, filter_size=None, N=None,nfmaps=None):
        self.ytest = None
        self.ytrain = None
        self.xtest = None
        self.model = None
        self.xtrain = None
        self.nlayers = nlayers
        self.filter_size = filter_size
        self.nfmaps = nfmaps
        self.N = N

    def make_input(self, xin, yin):
        """
        Take input of shape [nsamples,nvars,2] and implement cyclic padding to create input for CNN
        """

        offset = self.nlayers * (self.filter_size - 1) / 2
        offset = int(offset)
        nvars = self.N
        if len(xin.shape) > 2:
            nsamples = xin.shape[0]
            xf = xin[:,:,0]
            xstd = xin[:,:,1]
            y = yin
        else:
            nsamples = 1
            xf = xin[:,0]
            xstd = xin[:,1]
            xf = xf.reshape([1, nvars])
            xstd = xstd.reshape([1,nvars])
            y = yin.reshape([1,nvars])
        X = np.zeros([nsamples, nvars + 2 * offset, 3])
        X[:, offset:(nvars + offset), 0] = xf
        X[:, :offset, 0] = (xf[:, (nvars - offset):])
        X[:, (nvars + offset):, 0] = (xf[:, :offset])

        X[:, offset:(nvars + offset), 1] = y[:, :] - xf
        X[:, :offset, 1] = y[:, (nvars - offset):] - xf[:, (nvars - offset):]
        X[:, (nvars + offset):, 1] = y[:, :offset] - xf[:, :offset]

        X[:, offset:(nvars + offset), 2] = xstd
        X[:, :offset, 2] = (xstd[:, (nvars - offset):])
        X[:, (nvars + offset):, 2] = (xstd[:, :offset])
        return X

    def maketargets(self, x):
        """
        Create targets for CNN from analyses of shape [nvars,nsamples]
        """

        nsamples = x.shape[0]
        nvars = x.shape[1]
        ret = x.reshape([nsamples, nvars, 2])
        return ret

    def buildmodel(self):
        """
        Build keras model
        """

        nvars = self.N
        offset = self.nlayers * (self.filter_size - 1) / 2
        offset = int(offset)
        model = Sequential()
        model.add(Conv1D(self.nfmaps, self.filter_size, activation='relu', input_shape=(nvars + 2 * offset, 3)))
        for i in range(self.nlayers-2):
            model.add(Conv1D(self.nfmaps, self.filter_size, activation='relu'))
        model.add(Conv1D(2, self.filter_size, activation=None))
        self.model = model

    def train(self, training_fraction, nepochs, optimizer, experiment,tuning_iter = None):
        """
        Train keras model
        """

        self.model.compile(optimizer=optimizer, loss='mean_squared_error',
                           metrics=[keras.metrics.RootMeanSquaredError()])
        xa = experiment.xaens.data.mean(axis=1).transpose()[1:, :]
        xastd = experiment.xaens.data.std(axis=1).transpose()[1:, :]
        xf = experiment.xfens.data.mean(axis=1).transpose()[1:, :]
        xstd = experiment.xfens.data.std(axis=1).transpose()[1:, :]
        xx = experiment.xx.data[:, 1:].transpose()
        yy = experiment.yy.data[:, 1:].transpose()
        nsamples = xx.shape[0]
        nvars = xx.shape[1]
        ntrain = int(nsamples * training_fraction)
        xin = np.zeros([nsamples,nvars,2])
        xin[:,:,0] = xf
        xin[:,:,1] = xstd
        yout = np.zeros([nsamples,nvars,2])
        yout[:,:,0] = xa
        yout[:,:,1] = xastd
        xtrain = self.make_input(xin[:ntrain], yy[:ntrain, :])
        ytrain = self.maketargets(yout[:ntrain,:,:])
        xtest = self.make_input(xin[ntrain:], yy[ntrain:, :])
        ytest = self.maketargets(yout[ntrain:])
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        if tuning_iter is not None:
            fn = 'progress2/'+'iter_'+str(tuning_iter)
            model_checkpoint = ModelCheckpoint(fn + 'weights.{epoch:02d}.hdf5', save_weights_only=True)
        else:
            model_checkpoint = ModelCheckpoint('progress2/'+'weights.{epoch:02d}.hdf5',save_weights_only=True)
        res = self.model.fit(xtrain, ytrain, epochs=nepochs, validation_data=[xtest, ytest],callbacks=[model_checkpoint],verbose=2)
        return res

    def assimilate(self, x, y):
        """
        Assimilate a forecast and observation using the trained model
        """
        xin = self.make_input(x, y)
        yout = self.model.predict(xin).reshape([self.N,2])
        return yout