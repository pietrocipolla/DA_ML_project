def rms(xx,xa):
    ss = xx-xa
    ss = ss**2
    ss = ss.sum(axis=0)
    ss = ss/40
    ss = np.sqrt(ss)
    return(ss)

def res_to_rmse(fin,fout=None):
    with open(fin,'rb') as f:
        exp = pickle.load(f)
    rmse = rms(exp.ds.xx,exp.ds.xaens.mean(axis=1))
    rmse = rmse[rmse>0]
    if fout is not None:
        rmse.to_netcdf(fout)
    rmse = rmse.assign_coords({'yrs':('time',(5*rmse.time/365).data)})
    return(rmse)

def res_to_errs(fin,fout=None):
    with open(fin,'rb') as f:
        exp = pickle.load(f)

    err = exp.ds.xx-exp.ds.xaens.mean(axis=1)
    err = err/exp.ds.xaens.std(axis=1)
    err = -err
    err.to_netcdf(fout)