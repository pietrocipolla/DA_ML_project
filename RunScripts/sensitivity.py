import sys
sys.path.append('src')
sys.path.append('../src')
from src.Experiment_project import *
import pickle
import yaml


def runcase(settings_file):
    np.random.seed(0)
    with open(settings_file) as f:
        settings = yaml.safe_load(f)
    F = settings['F']
    N = settings['N']
    T = settings['T']
    x0 = F * np.ones(N)
    x0[0] = F + .01
    experiment = Experiment(settings=settings)
    experiment.assim_method = 'enkf'
    experiment.ds['xx'] = experiment.get_true(x0, T)
    r = experiment.ds.xx.std() * settings['r']
    experiment.r = r
    experiment.ds['yy'] = experiment.makeobs(r, x0, T)
    xf0 = experiment.make_ensemble(x0, r)
    experiment.assimilate(xf0)
    with open(settings['name']+'.pickle', 'wb') as f:
        pickle.dump(experiment, f)

fname = "./sensitivityrunsettings_s9.yml"
runcase(fname)