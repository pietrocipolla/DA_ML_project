import numpy as np
from scipy.integrate import odeint

'''Python class for ensemble and ML data assimilation'''

class DA:
    def __init__(self, M, h, sigma_z, sigma_x,gamma,loc = None):
        '''
        :param M: physical model object (class "Model")
        :param h: observation operator
        :param sigma_z: measurement standard deviation (float)
        :param sigma_x: model standard deviation (float)
        :param gamma: covariance inflation parameter
        :param loc: localization parameter (covariances further than distnace loc will be set to zero)
        '''
        self.M = M
        self.h = h
        self.r = sigma_z**2
        self.s = sigma_z
        self.sigma_x = sigma_x
        self.nvars = 0
        self.xf = 0
        self.n_ens = 0
        self.x0 = 0
        self.gamma = gamma
        self.loc = loc

    def initialize(self, x0, loc, scale, n_ens):
        '''
        Initialize an ensemble.

        :param x0: mean x (numpy array of length N)
        :param loc: localization parameter (int)
        :param scale: standard deviation of random noise to be added to x0 (float)
        :param n_ens: ensemble size (int)
        :return: none
        '''
        self.nvars = len(x0)
        self.n_ens = n_ens
        xi = np.zeros([n_ens, self.nvars])
        for i in range(n_ens):
            xi[i, :] = x0
        xi += np.random.normal(loc, scale, xi.shape)
        self.x0 = xi.transpose()
        self.xf = np.zeros(self.x0.shape)

    def forecast(self, t0, tf):
        '''
        Simulate ensemble forward from current x0
        :param t0: current time of x0
        :param tf: time to which ensemble should be simulated
        :return: none
        '''
        t = np.array([t0, tf])
        for i in range(self.n_ens):
            if len(self.x0.shape)>2:
                self.xf[:, i] = self.M(self.x0[:, i, 0], t)[-1, :]
            else:
                pass
            self.xf[:,i] = self.M(self.x0[:,i], t)[-1,:]

    def getK(self, hi):
        '''
        Calculate Kalman gain matrix

        :param hi: list of variables to be measured
        :return: kalman gain matrix
        '''
        cov = np.cov(self.x0)
        cov = cov*self.gamma

        if self.loc is not None:
            for i in range(self.nvars):
                for j in range(self.nvars):
                    dist = self.get_distance(i,j)
                    if dist>self.loc:
                        cov[i,j] = 0
        else:
            pass

        k = cov[hi, :][:, hi]
        for i in range(len(hi)):
            k[i,i] = k[i,i] + self.r
        k = np.linalg.inv(k)
        k = np.dot(cov[:, hi], k)
        return k

    def assim(self, xf, hi, z, k):
        '''
        Assimilate measurements

        :param xf: ensemble of forecasts
        :param hi: measurement indices
        :param z: measured value
        :param k: Kalman gain
        :return: analysis ensemble
        '''
        D = np.zeros([len(z), self.n_ens])
        for i in range(self.n_ens):
            #create measurement matrix with random noise added
            D[:, i] = z + np.random.normal(0, self.s, len(z))
        xa = xf + np.dot(k, D-xf[hi, :])
        return xa

    def assim_ann(self,nn,xfmean,z):
        x = np.zeros([self.nvars+6,2])
        x[3:self.nvars+3,0] = xfmean[:,0]
        x[0:3,0] = xfmean[self.nvars-3:,0]
        x[self.nvars+3:,0] = xfmean[:3,0]

        x[3:self.nvars+3,1] = z - xfmean[:,0]
        x[0:3,1] = z[self.nvars-3:] - xfmean[self.nvars-3:,0]
        x[self.nvars+3:,1] = z[:3] - xfmean[:3,0]

        x = x.reshape(1,x.shape[0],x.shape[1])
        y = nn.predict(x)
        return(y)


    def get_distance(self,i,j):
        mid = abs(i-j)
        if i>j:
            out = self.nvars-i
            out = out+j
        if i<j:
            out = self.nvars-j
            out = out+i
        if i == j:
            out = 0

        out = max([mid,out])
        return(out)