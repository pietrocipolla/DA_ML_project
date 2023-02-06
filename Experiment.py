import xarray as xr
import numpy as np
from L96 import *
from EnKF import *
from tqdm import tqdm


class Experiment:
    """
    Class for an enkf/ml-augmented experiment. Contains settings, results, and methods for creating/executing a run.

    Attributes
    ------------
    ds: xarray DataSet
        Contains time series data for the experiment including: truth, obs, ensemble forecast, and ensemble analysis
    N: int
        Number of Lorenz-96 variables
    F: float
        Lorenz-96 forcing parameter
    dt: float
        Time step between observations
    nens: int
        Ensemble size
    r: float
        Observation noise standard deviation
    gamma: float
        Covariance inflation factor
    loc: int
        Localization distance.
    frac: float
        Fraction of domain observed

    Methods
    ------------
    makeobs(std)
        Create synthetic observations
    get_true(x0,tf)
        Integrate L96 to create truth
    make_ensemble(x0,std)
        Create initial ensemble
    make_obs()
        Create obvservation operator consistent with self.frac
    assimilate(xf0,nn=None)
        assimilate observations

    """

    def __init__(self, ds=None, settings=None):
        if settings is None:
            settings = {}
        if ds is not None:
            self.ds = ds
        else:
            self.ds = xr.Dataset()
        try:
            self.N = settings['N']
        except KeyError:
            pass
        try:
            self.F = settings['F']
        except KeyError:
            pass
        try:
            self.dt = settings['dt']
        except KeyError:
            pass
        try:
            self.nens = settings['nens']
        except KeyError:
            pass
        try:
            self.r = settings['r']
        except KeyError:
            pass
        try:
            self.gamma = settings['gamma']
        except KeyError:
            pass
        try:
            self.loc = settings['loc']
        except KeyError:
            pass
        try:
            self.frac = settings['frac']
        except KeyError:
            pass

    def makeobs(self, std):
        """
        Create synthetic observations from truth (contained in ds.xx)

        :param std: observation noise standard deviation
        :return: xarray DataArray of synthetic observations
        """
        ret = self.ds.xx + np.random.normal(0, std, self.ds.xx.shape)
        return ret

    def get_true(self, x0, tf):
        """
        Create truth by integrating L96 forward.

        :param x0: initial state vector
        :param tf: time to integrate L96 to
        :return: xarray DataArray of true L96 trajectory
        """
        t = np.linspace(0, tf, int(tf / self.dt) + 1)
        M = L96(self.F, self.N)
        xx = M.integrate(x0, t)
        da = xr.DataArray(xx, coords=[list(range(self.N)), t], dims=['space', 'time'])
        return da

    def make_ensemble(self, x0, std):
        """
        Create initial ensemble.

        :param x0: Inital state vector
        :param std: Standard deviation of noise to add to create ensemble members
        :return: xarray DataArray of ensemble
        """
        x = np.zeros([self.N, self.nens])
        for i in range(self.N):
            x[i, :] = x0[i] + np.random.normal(0, std, self.nens)
        ret = xr.DataArray(x, coords=[list(range(self.N)), list(range(self.nens))], dims=['space', 'ensemble'])
        return ret

    def make_obs(self):
        """
        Create a observation operator with a random fraction of domain observed based on frac.

        :return: List of indices to be observed.
        """
        h = list(range(self.N))
        try:
            if self.frac < 1:
                np.random.shuffle(h)
                h = h[:int(self.N * self.frac)]
            else:
                pass
        except AttributeError:
            self.frac = 1
        return h

    def assimilate(self, xf0, nn=None):
        """
        assimilate observations contained in ds.yy. The first non NaN value of yy is assumed to be the same timestep as
        xf0, making the first assimilated observation the second element yy. Ensemble forecasts and analyses are
        both returned and saved within ds.

        :param xf0: Initial ensemble
        :param nn: NeuralNetwork object
        :return: (xaens,xfens), xarray DataArrays containing the ensemble analysis and ensemble forecast
        """

        # Create EnKF object for performaing assimilation
        enkf = EnKF(self.N, self.loc, self.gamma, self.nens)

        # Select only non NaN values of ds.yy to assimilate and count number of observations
        yy = self.ds.yy.isel(time=self.ds.yy.notnull()[0], space=self.ds.yy.space)
        nobs = self.ds.yy[0].count('time')
        nobs = int(nobs)
        nobs = nobs - 1
        xaens = np.zeros([self.N, self.nens, nobs + 1])
        xfens = np.zeros([self.N, self.nens, nobs + 1])
        xfens[:, :, 0] = xf0
        xaens[:, :, 0] = xf0
        xa = np.zeros([self.N, nobs + 1])
        for i in tqdm(range(1, nobs + 1), desc='assimilating: '):
            # For each observation
            for j in range(self.nens):
                # For each ensemble member

                # Integrate L96 from last analysis to current time step
                ret = self.get_true(xaens[:, j, i - 1], self.dt)
                xfens[:, j, i] = ret.isel(time=-1)
            y = yy[:, i]  # y  is the current observation
            if nn is None:
                # If method is EnKF, make observation operator:
                h = self.make_obs()

                # assimilate with forecast, observation, observation operator, and observation error:
                xaens[:, :, i] = enkf.ensemble_assim(xfens[:, :, i], y, h, self.r)
            else:
                if i % 2 == 0:
                    # If method is augmented, on even time steps use EnKF
                    h = self.make_obs()
                    xaens[:, :, i] = enkf.ensemble_assim(xfens[:, :, i], y, h, self.r)
                else:
                    # On odd time steps use NN

                    # Calculate vector analysis from forecast mean and observation
                    xpmean = nn.assimilate(xfens[:, :, i].mean(axis=1), y.values)
                    xa[:, i] = xpmean
                    # Adjust ensemble forecast so that the mean=analysis mean:
                    delta = xfens[:, :, i].mean(axis=1) - xpmean
                    for j in range(self.N):
                        xaens[j, :, i] = xfens[j, :, i] - delta[j]

        # Retrieve time array for non NaN observations and save results as DataArrays in ds
        t = self.ds.time.isel(time=self.ds.yy.notnull()[0])
        self.ds.coords['ensemble'] = list(range(self.nens))
        self.ds['xaens'] = xr.DataArray(xaens, coords=(self.ds.yy.space, list(range(self.nens)), t),
                                        dims=['space', 'ensemble', 'time'])
        self.ds['xfens'] = xr.DataArray(xfens, coords=(self.ds.yy.space, list(range(self.nens)), t),
                                        dims=['space', 'ensemble', 'time'])
        return xaens, xfens
