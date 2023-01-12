import unittest
import numpy as np
from L96 import *
from EnKF import *
from Experiment import *


class L96Test(unittest.TestCase):
    def testL96dt(self):
        '''
        Confirm dx/dt=0 when x0=F and dx/dt=F when x0=0
        '''
        F=8
        N=10
        t = 0
        x = F*np.ones(N)
        model = L96(F,N)
        dxdt = model.derivative(t,x)
        self.assertEqual(True,(dxdt==0).all())  # add assertion here
        x = np.zeros(N)
        dxdt = model.derivative(t,x)
        self.assertEqual(True,(dxdt==F).all())

    def testL96integration(self):
        '''
        Check integration by ensuring decay to x=F for F<3
        '''
        N = 10
        F = 0.5
        t0 = 0
        tf = 10
        dt = 0.05
        t = np.linspace(t0,tf,int((tf-t0)/dt)+1)
        x0 = np.ones(N)
        model = L96(F,N)
        xall = model.integrate(x0,t)
        self.assertEqual(((xall[:,-1]-xall[:,0])<.99).all(),True)

class ENKFTest(unittest.TestCase):
    def testH(self):
        '''
        Test observation operator returns correct entries. Observation operator is a list of indices.
        '''
        da = EnKF()
        h = [0,2,3,5]
        x = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]]).transpose()
        self.assertEqual(True,(da.obs(h,x)==(np.array([[1,3,4,6],[7,9,10,12]]).transpose())).all())

    def testK(self):
        '''
        Test calculation of Kalman gain matrix against exact analytical solutions
        '''

        #case1: r = 0, h identity, K=identity
        da = EnKF()
        cov = np.array([[1,.25],[.25,1]])
        r = 0
        h = [0,1]
        k = da.getK(cov,h,r)
        self.assertEqual(True,(np.identity(2)==k).all())

        #case 2: r=1, cov=I, K=0.5*I
        cov = np.array([[1,0],[0,1]])
        r = 1
        h = [0,1]
        k = da.getK(cov,h,r)
        self.assertEqual(True, (.5*np.identity(2) == k).all())

    def testLocalization(self):
        '''
        Test method for step function localization with cov=0 for distance>threshold
        '''
        N = 5
        localization = 1
        da = EnKF(N=N,loc=localization)
        cov = np.ones([5,5])
        exact = np.ones([5,5])
        exact[0,2:4] = 0
        exact[1,3:] = 0
        exact[2,0] = 0
        exact[2,4] = 0
        exact[3,:2] = 0
        exact[4,1:3] = 0
        localized = da.localize(cov)

        self.assertEqual(True,(localized==exact).all())

    def testAssim(self):
        '''
        Test assimilation for cov=I and r=1 against exact solution.
        '''
        N=2
        da = EnKF(N=N,loc=0)
        cov = np.array([[1,0],[0,1]])
        r = 1
        h = [0,1]
        xf = np.array([1,1])
        y = np.array([3,3])
        xp = da.assimilate(xf,y,h,cov,r)
        self.assertEqual(True, (xp==np.array([2,2])).all())

    def testEnsembleAssim(self):
        '''
        Test ensemble assimilation by checking that the ensemble mean and standard deviation
        match exact solution to within a tolerance.
        '''
        N=2
        nens = 10000
        da = EnKF(N=N,loc=0,nens=nens,gamma=1)
        r = 1
        h = [0,1]
        xf = np.ones([N,nens])
        y = np.array([3,3])
        np.random.seed(0)
        xf = xf+np.random.normal(0,1,[xf.shape[0],xf.shape[1]])
        xp = da.ensemble_assim(xf,y,h,r)
        self.assertLess(abs((xp.mean(axis=1)-np.array([2, 2])).mean()),.05)
        self.assertLess(abs(xp.std()**2-.5),.01)

    def testEnsembleInitialization(self):
        '''
        Test initialization of an ensemble given a initial condition
        '''
        N = 2
        nens = 10000
        da = EnKF(N=N,loc=0,nens=nens,gamma=1)
        std = 1
        np.random.seed(0)
        x0 = np.ones([N,nens])
        da.initialize(x0,std)
        self.assertLess(abs(da.x.std()-1),0.01)
        self.assertEqual(da.x.shape,(N,nens))

class ExperimentTest(unittest.TestCase):
    def testGetTruth(self):
        N = 10
        F = 0.5
        tf = 10
        dt = 0.05
        x0 = np.ones(N)
        experiment = Experiment(x0,N,F,dt)
        experiment.getTrue(tf)
        xall = experiment.xx
        self.assertEqual(((xall[:,-1]-xall[:,0])<.99).all(),True)

    def testMakeObs(self):
        N = 10
        F = 0.5
        dt = 0.05
        x0 = np.ones(10)
        experiment = Experiment(x0, N, F, dt)
        experiment.xx = np.ones([10,10000])
        experiment.Makeobs()
        self.assertLess(abs(1-experiment.yy.std()),0.01)

    def testAssimilate(self):
        N = 10
        F = 0.5
        tf = 10
        dt = 0.05
        nens = 25
        np.random.seed(0)
        x0 = np.random.normal(5,1,N)
        experiment = Experiment(x0,N,F,dt,nens=nens)
        experiment.getTrue(tf)
        experiment.Makeobs()
        experiment.Assimilate()
        self.assertLess(experiment.errors_a.std(),.5*experiment.errors_obs.std())

    def testRMS(self):
        N = 2
        F = 0.5
        dt = 0.05
        nens = 25
        x0 = np.ones(N)
        experiment = Experiment(x0,N,F,dt,nens=nens)
        experiment.errors_a = np.array([[1,2],[1,2]])
        expected = np.array([1,2])
        rms = experiment.rms()
        self.assertEqual(True,(expected==rms).all())






if __name__ == '__main__':
    unittest.main()
