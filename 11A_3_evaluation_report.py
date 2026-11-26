"""Report provisional scores and export unverified candidates separately."""
import argparse,json,csv
from pathlib import Path
from collections import Counter
from hydra_rules import normalize
from importlib import import_module
core=import_module('11A_3_hybrid_evaluator')
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path); p.add_argument('--smoke',action='store_true'); p.add_argument('--rules-only',action='store_true'); a=p.parse_args()
    out=a.run/('auto11a3_rules' if a.rules_only else 'auto11a3_smoke' if a.smoke else 'auto11a3_full'); meta=json.loads((out/'settings.json').read_text()); import hashlib
    for path,digest in meta['hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest: raise ValueError('Input changed: '+path)
    rows=core.read_jsonl(a.run/'responses.jsonl'); rows=rows[:meta['count']]; records=[json.loads((out/(r['id']+'.json')).read_text()) for r in rows if (out/(r['id']+'.json')).exists()]; lookup={r['id']:r for r in rows}
    def metrics(rs):
        scores=[r['score'] for r in rs if r['score'] is not None]; n=len(scores)
        return {'processed':len(rs),'accepted_provisional':n,'uncertain':len(rs)-n,'strict_accuracy_pct':100*sum(s['correct'] for s in scores)/n if n else None,'complete_pct':100*sum(s['complete'] for s in scores)/n if n else None,'unsupported_response_pct':100*sum(s['unsupported_count']>0 for s in scores)/n if n else None,'mean_rubric_score':sum(sum(s['criteria']) for s in scores)/n if n else None}
    report={'expected':meta['count'],'pending':meta['count']-len(records),'overall':metrics(records),'domains':{d:metrics([r for r in records if r['domain']==d]) for d in sorted({r['domain'] for r in records})},'flags':dict(Counter(f for r in records for f in r['flags'])),'candidate_counts':dict(Counter(r['candidate_label'] for r in records if r['candidate_label'])),'warning':'Self-judge provisional results; accepted subset is not an unbiased estimate of full benchmark accuracy.'}
    core.write(out/'report.json',report)
    candidates=[]
    if len(records)==40 and not a.smoke and not a.rules_only:
        for r in records:
            if r['candidate_label']:
                candidates.append(dict(lookup[r['id']],review_label=r['candidate_label'],unsupported_quote=' | '.join(r['score']['unsupported_quotes']),review_notes=r['score']['notes'],review_status='UNVERIFIED_SELF_JUDGE'))
    (out/'11C_provisional_candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in candidates))
    with (out/'verification_queue.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['id','domain','question','answer','criteria','flags','provisional_score','candidate_label'])
        for r in sorted(records,key=lambda x:(not bool(x['flags']),x['id'])):
            orig=lookup[r['id']]; w.writerow([r['id'],r['domain'],orig['question'],orig['answer'],json.dumps(orig['criteria']),'; '.join(r['flags']),json.dumps(r['score']),r['candidate_label']])
    print(json.dumps(report,indent=2)); print('Provisional candidates:',len(candidates)); print('Do not use for 11C until verified.')
if __name__=='__main__': main()
