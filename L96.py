import numpy as np
from scipy.integrate import solve_ivp

class L96:
    def __init__(self,F,N):
        self.F = F
        self.N = N
    def derivative(self,t,x):
        '''
        Calculate the derivative of x

        :param x: current value of x (numpy array of length N)
        :param t: current time (not used, d/dt doesn't have explicit time dependence
        :return: derivative of x (numpy array of length N)
        '''
        d = np.zeros(self.N)
        for i in range(self.N):
            d[i] = (x[(i + 1) % self.N] - x[i - 2]) * x[i - 1] - x[i] + self.F
        return(d)

    def integrate(self,x0,t):
        t0 = t[0]
        tf = t[-1]
        ret = solve_ivp(self.derivative,[t0,tf],x0,t_eval = t)

        return(ret['y'])