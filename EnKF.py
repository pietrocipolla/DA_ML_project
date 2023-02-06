import numpy as np


class EnKF:
    """
    Class for implementing the Ensemble Kalman Filter.

    Attributes
    ------------
    nvars: int
        Number of variables in state vector
    loc: int
        Localization distance
    gamma: float
        Covariance inflation factor
    nens:
        Ensemble size
    """

    def __init__(self, N=None, loc=None, gamma=1, nens=1):
        self.nvars = N
        self.loc = loc
        self.gamma = gamma
        self.nens = nens

    def obs(self, h, x):
        """
        Execute observation operator

        :param h: list of variable indices to be observed
        :param x: state vector
        :return: h(x)
        """

        ret = x[list(h)]
        return (ret)

    def getK(self, cov, h, r):
        """
        Compute Kalman gain matrix

        :param cov: forecast covariance
        :param h: list of variables to be observed
        :param r: observation standard deviation
        :return: kalman gain matrix
        """
        k = self.obs(h, cov.transpose())
        k = k.transpose()
        second_term = self.obs(h, k)
        variance = r**2
        for i in range(len(h)):
            second_term[i, i] = second_term[i, i] + variance
        second_term = np.linalg.inv(second_term)
        k = np.dot(k, second_term)
        return (k)

    def localize(self, cov):
        """
        Localize the covariance matrix according to self.loc. If distance>loc, set cov=0.
        :param cov: covariance matrix to localize
        :return: localized covariance matrix
        """
        loc = self.loc
        for i in range(self.nvars):
            for j in range(self.nvars):
                mid = abs(i - j)
                if i > j:
                    out = self.nvars - i
                    out = out + j
                if i < j:
                    out = self.nvars - j
                    out = out + i
                if i == j:
                    out = 0
                dist = min([mid, out])
                if dist > loc:
                    cov[i, j] = 0
        return (cov)

    def assimilate(self, xf, y, h, cov, r):
        """
        assimilate an observation.

        :param xf: forecast
        :param y: observation
        :param h: observation operator list
        :param cov: forecast covariance
        :param r: observation error standard deviation
        :return: posterior estimate of x
        """
        cov = self.localize(cov)
        cov = cov * self.gamma
        k = self.getK(cov, h, r)
        xp = xf + np.dot(k, self.obs(h, y) - self.obs(h, xf))
        return (xp)

    def ensemble_assim(self, xf, y, h, r):
        """
        Perform ensemble assimilation.

        :param xf: Forecast ensemble
        :param y: Observation vector
        :param h: observation operator list
        :param r: observation error standard deviation
        :return: posterior ensemble
        """
        cov = np.cov(xf)
        D = np.zeros([self.nvars, self.nens])
        for i in range(self.nvars):
            D[i, :] = y[i].values + np.random.normal(0, r, self.nens)
        xp = self.assimilate(xf, D, h, cov, r)
        return (xp)
