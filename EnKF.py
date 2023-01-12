import numpy as np


class EnKF:
    def __init__(self,N=None,loc=None,gamma=1,nens=1):
        self.nvars = N
        self.loc = loc
        self.gamma = gamma
        self.nens = nens
    def obs(self,h,x):
        ret = x[list(h)]
        return(ret)

    def getK(self,cov,h,r):
        k = self.obs(h,cov.transpose())
        k = k.transpose()
        second_term = self.obs(h,k)
        nvars = cov.shape[0]
        for i in range(nvars):
            second_term[i,i] = second_term[i,i]+r
        second_term = np.linalg.inv(second_term)
        k = np.dot(k,second_term)
        return(k)

    def localize(self,cov):
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
                if dist>loc:
                    cov[i,j] = 0
        return(cov)

    def assimilate(self,xf,y,h,cov,r):
        cov = self.localize(cov)
        cov = cov*self.gamma
        k = self.getK(cov,h,r)
        xp = xf+np.dot(k,y-self.obs(h,xf))
        return(xp)

    def ensemble_assim(self,xf,y,h,r):
        cov = np.cov(xf)
        D = np.zeros([self.nvars,self.nens])
        for i in range(self.nvars):
            D[i,:] = y[i]+np.random.normal(0,r,self.nens)
        xp = self.assimilate(xf,D,h,cov,r)
        return(xp)

    def initialize(self,x0,std):
        x = np.zeros([self.nvars,self.nens])
        for i in range(self.nvars):
            x[i,:] = x0[i]+np.random.normal(0,std,self.nens)
        self.x = x