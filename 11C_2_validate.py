import argparse, hashlib, json
from pathlib import Path
import numpy as np
DEFAULT='Experiment11/results/contrast_20261127_112213_834928000'
def require(ok,msg):
    if not ok: raise ValueError(msg)
def validate(run,dataset):
    run=Path(run); raw=Path(dataset).read_bytes(); pairs=json.loads(raw); report=json.loads((run/'report.json').read_text())
    require(report['pass'] is True,'Collection did not pass')
    require(hashlib.sha256(raw).hexdigest()==report['dataset_sha256'],'Dataset fingerprint mismatch')
    require(len(pairs)==10 and len({p['id'] for p in pairs})==10,'Expected ten unique pairs')
    require(sorted(p['split'] for p in pairs)==['direction_check']*5+['direction_dev']*5,'Invalid splits')
    require(report['layers']==[0,20,41],'Unexpected layers')
    entries={p['id']:p for p in report['pairs']}; require(len(report['pairs'])==10 and set(entries)=={p['id'] for p in pairs},'Report pair mismatch')
    expected={f"{p['id']}_{b}_L{l}.npz" for p in pairs for b in ('correct','incorrect') for l in report['layers']}
    require({p.name for p in run.glob('*.npz')}==expected,'Missing or extra NPZ files')
    data={}
    for p in pairs:
        pid=p['id']; e=entries[pid]; require(e['split']==p['split'] and e['common_prefix_check'] is True,pid+' metadata mismatch')
        branches={}; data[pid]={}
        for b in ('correct','incorrect'):
            m=e['branches'][b]; t=np.asarray(m['token_ids']); n=m['prefix_length']; pos=np.arange(n-1,len(t))
            require(t.ndim==1 and t.dtype.kind in 'iu' and np.all(t>=0) and 0<n<len(t)<=256 and m['captured_rows']==len(pos) and m['continuation']==p[b],pid+' invalid branch metadata')
            require(np.isfinite(m['max_logit_difference']) and m['max_logit_difference']>=0,pid+' invalid logit difference')
            branches[b]=(t,n); data[pid][b]={}
            for l in report['layers']:
                with np.load(run/f'{pid}_{b}_L{l}.npz',allow_pickle=False) as z:
                    require(set(z.files)=={'inputs','outputs','positions','token_ids'},'Unexpected arrays')
                    x,y=z['inputs'],z['outputs']; pp,tt=z['positions'],z['token_ids']
                    require(x.shape==(len(pos),10240) and y.shape==(len(pos),2560) and x.dtype==y.dtype==np.float32 and np.isfinite(x).all() and np.isfinite(y).all(),pid+' shape/dtype/finite failure')
                    require(pp.dtype.kind in 'iu' and tt.dtype.kind in 'iu' and np.array_equal(pp,pos) and np.array_equal(tt,t[pos]),pid+' alignment failure')
                    data[pid][b][l]={'anchor_x':x[0].copy(),'anchor_y':y[0].copy(),'y':y[1:].copy()}
        t,n=branches['correct']; u,k=branches['incorrect']; require(n==k and np.array_equal(t[:n],u[:k]),pid+' prefix mismatch')
        for l in report['layers']:
            for key in ('anchor_x','anchor_y'):
                require(np.allclose(data[pid]['correct'][l][key],data[pid]['incorrect'][l][key],rtol=1e-4,atol=1e-4),pid+' anchor mismatch')
        print(pid,'PASS')
    print('Integrity PASS: 60 files, 10 pairs, 5 dev / 5 check')
    return pairs,data,report
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run',nargs='?',default=DEFAULT); p.add_argument('--dataset',default=str(Path(__file__).with_name('hydra_contrast_pairs.json'))); a=p.parse_args(); validate(a.run,a.dataset)
