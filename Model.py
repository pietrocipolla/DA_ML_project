from scipy.integrate import odeint
import numpy as np

class Model:
    '''
    Physical model class. Currently only supports Lorenz96
    '''
    def __init__(self, N, F):
        '''
        :param N: number of lorenz variables
        :param F: forcing parameter
        '''
        self.F = F
        self.N = int(N)

    def L96(self, x, t):
        '''
        Calculate the derivative of x

        :param x: current value of x (numpy array of length N)
        :param t: current time (not used, d/dt doens't have explicit time dependence
        :return: derivative of x (numpy array of length N)
        '''
        d = np.zeros(self.N)
        for i in range(self.N):
            d[i] = (x[(i + 1) % self.N] - x[i - 2]) * x[i - 1] - x[i] + self.F
        return d

    def M(self, x0, t, noise=False):
        '''
        Simulate lorenz96 system forward from initial state

        :param x0: starting value of x (numpy array of length N)
        :param t: array of times at which x should be evaluated
        :param noise: Boolean, whether to add model error noise (optional)
        :return: numpy array of x values at times specified in t (dimension [ntimes,Nx])
        '''
        ret = odeint(self.L96, x0, t)
        if not noise:
            pass
        else:
            ret[-1, :] += np.random.normal(loc=0, scale=noise, size=self.N)
        return ret