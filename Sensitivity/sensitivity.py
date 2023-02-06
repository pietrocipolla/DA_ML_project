import numpy as np
from Experiment import *
import pickle


def runcase(case):
    np.random.seed(0)
    F = 8
    N = 40
    dt = 0.05
    T = 2000
    frac = 1
    cases = [
        {'N': N, 'F': F, 'nens': 100, 'loc': 5, 'dt': dt, 'frac': frac, 'gamma': 1},
        {'N': N, 'F': F, 'nens': 100, 'loc': 5, 'dt': dt, 'frac': frac, 'gamma': 1},
        {'N': N, 'F': F, 'nens': 100, 'loc': 5, 'dt': dt, 'frac': frac, 'gamma': 1},
        {'N': N, 'F': F, 'nens': 50, 'loc': 5, 'dt': dt, 'frac': frac, 'gamma': 1},
        {'N': N, 'F': F, 'nens': 1000, 'loc': 5, 'dt': dt, 'frac': frac, 'gamma': 1},
        {'N': N, 'F': F, 'nens': 100, 'loc': 5, 'dt': dt, 'frac': frac, 'gamma': 1.01},
        {'N': N, 'F': F, 'nens': 100, 'loc': 5, 'dt': dt, 'frac': frac, 'gamma': 1.05},
        {'N': N, 'F': F, 'nens': 100, 'loc': 5, 'dt': dt, 'frac': frac, 'gamma': 1.1},
        {'N': N, 'F': F, 'nens': 100, 'loc': 3, 'dt': dt, 'frac': frac, 'gamma': 1},
        {'N': N, 'F': F, 'nens': 100, 'loc': 7, 'dt': dt, 'frac': frac, 'gamma': 1},
    ]
    casenames = ['base', 's1', 's2', 's3_50', 's4', 's5', 's6', 's7', 's8', 's9']
    x0 = F * np.ones(N)
    x0[0] = F + .01
    # settings={'N':N,'F':F,'nens':nens,'loc':loc,'dt':dt,'frac':frac,'gamma':gamma}
    settings = cases[case]
    experiment = Experiment(settings=settings)
    experiment.ds['xx'] = experiment.get_true(x0, T)
    if case == 1:
        r = experiment.ds.xx.std() * .2
    elif case == 2:
        r = experiment.ds.xx.std() * .4
    else:
        r = experiment.ds.xx.std() * .3
    experiment.r = r
    experiment.ds['yy'] = experiment.makeobs(r)
    xf0 = experiment.make_ensemble(x0, r)
    experiment.assimilate(xf0)
    with open(casenames[case]+'.pickle', 'wb') as f:
        pickle.dump(experiment, f,protocol=4)


ret = runcase(4)