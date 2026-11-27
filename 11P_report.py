import argparse,csv,json,statistics as s,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def main():
 p=argparse.ArgumentParser(); p.add_argument('run',type=Path); a=p.parse_args(); out=a.run; state=json.loads((out/'checkpoint.json').read_text()); rows=state['rows']; expected=5 if state['meta']['smoke'] else 300
 if len(rows)!=expected: raise RuntimeError(f'Incomplete: {len(rows)}/{expected}')
 write(out/'responses.csv',rows); summary=[]
 for k in ['P0','P1','P2','P3','P4']:
  rr=[r for r in rows if r['condition']==k]; times=sorted(r['seconds'] for r in rr); first=[r['ttft'] for r in rr if r['ttft'] is not None]; base={(r['id'],r['rep']):r for r in rows if r['condition']=='P0'}
  summary.append(dict(condition=k,n=len(rr),median_seconds=s.median(times),p95_seconds=times[math.ceil(.95*len(times))-1],median_first_text_seconds=s.median(first) if first else None,first_text_observed=len(first),mean_end_to_end_tps=s.mean(r['tps'] for r in rr),mean_tokens=s.mean(r['tokens'] for r in rr),max_mlx_peak_mib=max(r['peak_mib'] for r in rr),accuracy_pct=100*s.mean(r['correct'] for r in rr),cap_hits=sum(r['cap_hit'] for r in rr),paired_median_latency_reduction_pct=s.median(100*(1-r['seconds']/base[r['id'],r['rep']]['seconds']) for r in rr)))
 b=summary[0]
 for r in summary: r.update(median_latency_reduction_pct=100*(1-r['median_seconds']/b['median_seconds']),accuracy_change_pp=r['accuracy_pct']-b['accuracy_pct'])
 write(out/'summary.csv',summary); lines=['HYDRA 11P — '+('SMOKE ONLY' if expected==5 else 'PERFORMANCE BENCHMARK'),f'Completed {len(rows)}/{expected}. Objective choice accuracy only; semantic review pending.','TTFT proxy = first nonempty streamed text; throughput includes prefill; MLX peak is active allocator memory, not system RAM. P4 tests prefill chunk size, not guaranteed acceleration. Repetitions are not independent questions.']
 for r in summary: lines.append('\n'+r['condition']+'\n'+'\n'.join(f'{k}: {v:.4f}' if isinstance(v,float) else f'{k}: {v}' for k,v in r.items() if k!='condition'))
 (out/'report.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
 review=out/'semantic_review.csv'
 if not review.exists():
  cases={c['id']:c for c in json.loads((ROOT/'hydra_11p_20.json').read_text())}; write(review,[dict(id=r['id'],rep=r['rep'],condition=r['condition'],question=cases[r['id']]['question'],expected=cases[r['id']]['correct'],rubric=cases[r['id']]['rubric'],answer=r['answer'],semantic_correct='',complete='',unsupported_count='',notes='') for r in rows])
if __name__=='__main__': main()
