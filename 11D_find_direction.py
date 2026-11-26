"""Fit a matched mean-difference direction and evaluate on separate pairs."""
import argparse
import numpy as np

def find_failure_direction(productive,failed):
    p,f=map(lambda x:np.asarray(x,dtype=np.float32),(productive,failed))
    if p.ndim!=2 or p.shape!=f.shape or len(p)<2 or not np.isfinite(p).all() or not np.isfinite(f).all(): raise ValueError('Need finite matched activation arrays with at least two pairs')
    d=(f-p).mean(axis=0); norm=np.linalg.norm(d)
    if norm<1e-8: raise ValueError('No usable direction')
    return d/norm
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('train'); p.add_argument('validation'); p.add_argument('output'); a=p.parse_args()
    with np.load(a.train,allow_pickle=False) as t: u=find_failure_direction(t['productive'],t['failed'])
    with np.load(a.validation,allow_pickle=False) as v:
        good,bad=v['productive'],v['failed']; assert good.shape==bad.shape and good.ndim==2 and len(good)>0
        print('Validation paired separation:',np.mean((bad-good)@u>0))
    np.save(a.output,u); print('Candidate direction saved; correlation is not causation.')
