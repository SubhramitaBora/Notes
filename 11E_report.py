import argparse,csv,json,re
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); a=p.parse_args(); out=a.run
    state=json.loads((out/'checkpoint.json').read_text()); rows=state['rows']; base={r['id']:r for r in rows if r['condition']=='baseline'}
    for r in rows:
        b=base.get(r['id']); r['margin_change']=r['margin']-b['margin'] if b else None
        norm=lambda s:re.sub(r'[^a-z0-9]+',' ',s.lower()).strip()
        r['exact_correct']=norm(r['generated'])==norm(r['expected']); r['exact_incorrect']=norm(r['generated'])==norm(r['incorrect']); r['review_required']=not (r['exact_correct'] or r['exact_incorrect'])
    summaries=[]
    for name,_,_ in state['meta']['conditions']:
        for kind in ('target','regression'):
            group=[r for r in rows if r['condition']==name and r['kind']==kind and r['margin_change'] is not None]
            if not group: continue
            summaries.append(dict(condition=name,kind=kind,n=len(group),mean_margin=float(np.mean([r['margin'] for r in group])),mean_change=float(np.mean([r['margin_change'] for r in group])),improved=sum(r['margin_change']>1e-6 for r in group),correct_preferred=sum(r['margin']>0 for r in group),exact_correct=sum(r['exact_correct'] for r in group),exact_incorrect=sum(r['exact_incorrect'] for r in group),review_required=sum(r['review_required'] for r in group),cap_hits=sum(r['possibly_truncated'] for r in group)))
    for filename,data in [('responses.csv',rows),('summary.csv',summaries)]:
        if data:
            with (out/filename).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    validation=json.loads((out/'validation.json').read_text()) if (out/'validation.json').exists() else None
    lines=['HYDRA 11E — exploratory causal intervention',f'Validation: {validation}','Mean token log-likelihood margins in nats; positive favors correct. Exact generation matches are conservative proxies, not semantic accuracy.','No automatic weight-edit approval. Compare target effects with reverse/random controls and regression results.']+[str(s) for s in summaries]
    (out/'report.txt').write_text('\n'.join(lines)+'\n'); (out/'report.json').write_text(json.dumps({'validation':validation,'summary':summaries},indent=2)); print('\n'.join(lines))
if __name__=='__main__': main()
