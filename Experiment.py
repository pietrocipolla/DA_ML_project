from L96 import *
from EnKF import *
from tqdm import tqdm
class Experiment():
    def __init__(self,x0,N,F,deltat,xstd=1,ystd = 1,loc=0,gamma=1,nens=1):
        self.x0 = x0
        self.N = N
        self.F = F
        self.M = L96(F, N)
        self.deltat = deltat
        self.ystd = ystd
        self.xstd = xstd
        self.deltat = deltat
        self.da = EnKF(N,loc,gamma,nens)
        self.da.initialize(x0,xstd)

    def getTrue(self,T):
        self.xx = self.M.integrate(self.x0,np.linspace(0,T,int(T/self.deltat+1)))

    def Makeobs(self):
        self.yy = self.xx+np.random.normal(0,self.ystd,self.xx.shape)

    def Assimilate(self):
        nobs = self.yy.shape[1]-1
        ti = [0,self.deltat]
        r = self.ystd ** 2
        xf = np.zeros([self.N,self.da.nens])
        xa = np.zeros([nobs,self.N,self.da.nens])
        xfall = np.zeros(xa.shape)
        for i in tqdm(range(nobs),desc='assimilating: '):
            yi = self.yy[:,i+1]
            for j in range(self.da.nens):
                xf[:, j] = self.M.integrate(self.da.x[:, j], ti)[:, 1]
            xfall[i] = xf
            h = list(range(self.N))
            xp = self.da.ensemble_assim(xf, yi, h, r)
            self.da.x = xp
            xa[i] = xp
        self.xa = xa
        self.xf = xfall
        self.errors_a = xa.mean(axis=2).transpose()-self.xx[:,1:]
        self.errors_obs = self.yy-self.xx

    def rms(self):
        ss = (self.errors_a**2).sum(axis=0)
        ss = ss/self.N
        ret = np.sqrt(ss)
        return(ret)

