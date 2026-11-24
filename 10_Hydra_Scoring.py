import argparse,csv,json,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(); p.add_argument('run',type=Path); a=p.parse_args()
settings=json.loads((a.run/'settings.json').read_text()); data=ROOT/'benchmarks/hydra_100.json'
if hashlib.sha256(data.read_bytes()).hexdigest()!=settings['dataset_sha256']: raise SystemExit('Dataset changed')
cases={c['id']:c for c in json.loads(data.read_text())}; rows=list(csv.DictReader((a.run/'results.csv').open())); expected={(i,v) for i in settings['case_ids'] for v in settings['variants']}
if len(rows)!=len(expected) or {(r['case_id'],r['variant']) for r in rows}!=expected: raise SystemExit('Incomplete or duplicate run; do not score')
metrics=[]
for v in settings['variants']:
    group=[r for r in rows if r['variant']==v]; n=len(group); evidence=[r for r in group if cases[r['case_id']]['evidence_case']]; valid=[r for r in group if r['choice'] in 'ABCD' and len(r['choice'])==1]; correct=sum(r['choice']==cases[r['case_id']]['answer'] for r in group); wrong_evidence=sum(r['choice']!=cases[r['case_id']]['answer'] for r in evidence if r in valid)
    metrics.append(dict(variant=v,n=n,correct=correct,accuracy=100*correct/n,invalid=n-len(valid),evidence_n=len(evidence),evidence_wrong_choices=wrong_evidence,evidence_invalid=sum(r not in valid for r in evidence),mean_seconds=statistics.mean(float(r['seconds']) for r in group)))
base=metrics[0]; lines=[f'HYDRA-100 {settings["stage"].upper()}','Wrong evidence choices are a proxy, not a free-text hallucination rate. Invalid answers count as incorrect.']
for m in metrics:
    m['accuracy_change_pp']=m['accuracy']-base['accuracy']; lines.append(f'{m["variant"]}: {m["correct"]}/{m["n"]} correct ({m["accuracy"]:.2f}%), change {m["accuracy_change_pp"]:+.2f} pp; invalid={m["invalid"]}; evidence wrong={m["evidence_wrong_choices"]}/{m["evidence_n"]}, evidence invalid={m["evidence_invalid"]}; latency={m["mean_seconds"]:.3f}s')
if settings['stage']=='dev':
    winner=max(metrics[1:],key=lambda m:(m['correct'],-m['evidence_wrong_choices']-m['evidence_invalid']))
    lines.append('Development winner: '+winner['variant'] if winner['correct']>base['correct'] else 'No candidate improves development accuracy; do not claim a winner.')
report='\n'.join(lines); print(report); (a.run/'report.txt').write_text(report+'\n'); (a.run/'metrics.json').write_text(json.dumps(metrics,indent=2))
