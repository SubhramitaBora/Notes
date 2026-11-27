import argparse,csv,json,importlib,datetime,re,hashlib
from pathlib import Path
import numpy as np
vmod=importlib.import_module('11C_2_validate')
def unit(x):
    n=np.linalg.norm(x)
    if not np.isfinite(n) or n<1e-12: raise ValueError('Degenerate direction')
    return x/n
def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
def words(s): return set(re.findall(r'\b\w+\b',s.lower()))
def main():
    p=argparse.ArgumentParser(); p.add_argument('run',nargs='?',default=vmod.DEFAULT); p.add_argument('--dataset',default=str(Path(__file__).with_name('hydra_contrast_pairs.json'))); a=p.parse_args()
    pairs,data,report=vmod.validate(a.run,a.dataset); dev=[i for i,p in enumerate(pairs) if p['split']=='direction_dev']; check=[i for i,p in enumerate(pairs) if p['split']=='direction_check']
    out=Path(a.run)/('direction_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')); out.mkdir(); rows=[]; summaries=[]
    for l in report['layers']:
        c=[data[p['id']]['correct'][l]['y'].astype(np.float64) for p in pairs]; w=[data[p['id']]['incorrect'][l]['y'].astype(np.float64) for p in pairs]
        cm=np.stack([x.mean(0) for x in c]); wm=np.stack([x.mean(0) for x in w]); d=wm-cm; direction=unit(d[dev].mean(0)); center=(cm[dev].mean(0)+wm[dev].mean(0))/2
        # Equal token-count prefix and final-position sensitivity checks, not semantic alignment.
        matched=np.stack([y[:min(len(x),len(y))].mean(0)-x[:min(len(x),len(y))].mean(0) for x,y in zip(c,w)])
        final=np.stack([y[-1]-x[-1] for x,y in zip(c,w)])
        mv=unit(matched[dev].mean(0)); fv=unit(final[dev].mean(0)); gaps=d@direction; loo=[]; stability=[]
        for i in dev:
            z=unit(d[[j for j in dev if j!=i]].mean(0)); loo.append(float(d[i]@z)); stability.append(float(z@direction))
        lexical=[]; lengths=[]; certainty=[]; neg=[]
        for i,pair in enumerate(pairs):
            x,y=words(pair['correct']),words(pair['incorrect']); overlap=len(x&y)/len(x|y); lexical.append(overlap); lengths.append(len(w[i])-len(c[i])); certainty.append(int(bool(y&{'proves','definitely','always'}))-int(bool(x&{'proves','definitely','always'}))); neg.append(int(bool(y&{'not','no','never'}))-int(bool(x&{'not','no','never'})))
            rows.append(dict(layer=l,id=pair['id'],split=pair['split'],correct_tokens=len(c[i]),incorrect_tokens=len(w[i]),correct_projection=float((cm[i]-center)@direction),incorrect_projection=float((wm[i]-center)@direction),gap=float(gaps[i]),pair_cosine=float(d[i]@direction/max(np.linalg.norm(d[i]),1e-12)),loo_gap=loo[dev.index(i)] if i in dev else '',matched_gap=float(matched[i]@mv),final_gap=float(final[i]@fv),lexical_jaccard=overlap,certainty_difference=certainty[-1],negation_difference=neg[-1]))
        s=dict(layer=l,dev_positive=int(sum(gaps[dev]>0)),check_positive=int(sum(gaps[check]>0)),loo_positive=int(sum(np.array(loo)>0)),check_mean_gap=float(gaps[check].mean()),check_median_gap=float(np.median(gaps[check])),loo_mean_cosine=float(np.mean(stability)),matched_direction_cosine=float(mv@direction),final_direction_cosine=float(fv@direction),matched_check_positive=int(sum(matched[check]@mv>0)),final_check_positive=int(sum(final[check]@fv>0)),check_gap_length_corr=corr(gaps[check],np.array(lengths)[check]),check_gap_lexical_corr=corr(gaps[check],np.array(lexical)[check]),check_gap_certainty_corr=corr(gaps[check],np.array(certainty)[check]),check_gap_negation_corr=corr(gaps[check],np.array(neg)[check]))
        summaries.append(s); np.savez_compressed(out/f'candidate_L{l}.npz',direction=direction.astype(np.float32),center=center.astype(np.float32),matched_direction=mv.astype(np.float32),final_direction=fv.astype(np.float32))
    for name,records in [('pair_scores',rows),('layer_comparison',summaries)]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    meta=dict(source=str(Path(a.run).resolve()),dataset_sha256=report['dataset_sha256'],source_report_sha256=hashlib.sha256((Path(a.run)/'report.json').read_bytes()).hexdigest(),dev_ids=[pairs[i]['id'] for i in dev],check_ids=[pairs[i]['id'] for i in check],scope='Constructed continuations; exploratory directions only; no causal evidence. Check split is preliminary, not final held-out validation.',layers=summaries)
    (out/'report.json').write_text(json.dumps(meta,indent=2,allow_nan=False)); lines=['HYDRA 11D — EXPLORATORY DIRECTIONS',meta['scope'],'Positive gap means incorrect projects higher than correct. All counts out of five.','Correlations on five pairs are descriptive; lexical and position checks do not remove confounding.','No automatic layer selection or weight editing.']
    for s in summaries: lines.extend(['',f"Layer {s['layer']}"]+[f'  {k}: {v}' for k,v in s.items() if k!='layer'])
    (out/'report.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines)); print('Output:',out)
if __name__=='__main__': main()
