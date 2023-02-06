import keras
import numpy as np
from keras import Sequential
from keras.layers import *


class NeuralNet:
    def __init__(self, nlayers=None, filter_size=None, N=None):
        self.ytest = None
        self.ytrain = None
        self.xtest = None
        self.model = None
        self.xtrain = None
        self.nlayers = nlayers
        self.filter_size = filter_size
        self.N = N

    def make_input(self, xin, yin):
        """
        Take input of shape [nsamples,nvars] and implement cyclic padding to create input for CNN
        """

        offset = self.nlayers * (self.filter_size - 1) / 2
        offset = int(offset)
        nvars = self.N
        if len(xin.shape) > 1:
            nsamples = xin.shape[0]
            xf = xin
            y = yin
        else:
            nsamples = 1
            xf = xin.reshape([1, len(xin)])
            y = yin.reshape(xf.shape)
        X = np.zeros([nsamples, nvars + 2 * offset, 2])
        X[:, offset:(nvars + offset), 0] = xf
        X[:, :offset, 0] = (xf[:, (nvars - offset):])
        X[:, (nvars + offset):, 0] = (xf[:, :offset])

        X[:, offset:(nvars + offset), 1] = y[:, :] - xf
        X[:, :offset, 1] = y[:, (nvars - offset):] - xf[:, (nvars - offset):]
        X[:, (nvars + offset):, 1] = y[:, :offset] - xf[:, :offset]
        return X

    def maketargets(self, x):
        """
        Create targets for CNN from analyses of shape [nvars,nsamples]
        """

        nsamples = x.shape[0]
        nvars = x.shape[1]
        ret = x.reshape([nsamples, nvars, 1])
        return ret

    def buildmodel(self):
        """
        Build keras model
        """

        nvars = self.N
        offset = self.nlayers * (self.filter_size - 1) / 2
        offset = int(offset)
        model = Sequential()
        model.add(Conv1D(5, 3, activation='relu', input_shape=(nvars + 2 * offset, 2)))
        model.add(Conv1D(5, 3, activation='relu'))
        model.add(Conv1D(1, 3, activation=None))
        self.model = model

    def train(self, training_fraction, nepochs, optimizer, experiment):
        """
        Train keras model
        """

        self.model.compile(optimizer=optimizer, loss='mean_squared_error',
                           metrics=[keras.metrics.RootMeanSquaredError()])
        xa = experiment.xaens.data.mean(axis=1).transpose()[1:, :]
        xf = experiment.xfens.data.mean(axis=1).transpose()[1:, :]
        xx = experiment.xx.data[:, 1:].transpose()
        yy = experiment.yy.data[:, 1:].transpose()
        nsamples = xx.shape[0]
        ntrain = int(nsamples * training_fraction)
        xtrain = self.make_input(xf[:ntrain, :], yy[:ntrain, :])
        ytrain = self.maketargets(xa[:ntrain, :])
        xtest = self.make_input(xf[ntrain:, :], yy[ntrain:, :])
        ytest = self.maketargets(xx[ntrain:, :])
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        res = self.model.fit(xtrain, ytrain, epochs=nepochs, validation_data=[xtest, ytest])
        return res

    def assimilate(self, xf, y):
        """
        Assimilate a forecast and observation using the trained model
        """
        xin = self.make_input(xf, y)
        yout = self.model.predict(xin).reshape(self.N)
        return yout
