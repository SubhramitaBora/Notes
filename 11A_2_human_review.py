"""Prepare local review CSV, then score human labels and export 11C candidates."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
FIELDS=['criterion_1','criterion_2','criterion_3','correct','complete','unsupported_count','unsupported_quote','failure_type','notes']
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['prepare','score']); p.add_argument('run',type=Path); a=p.parse_args(); run=a.run; review=run/'human_review_11A2.csv'
 old=(run/'comparison.json').exists(); source=run/('comparison.json' if old else 'responses.jsonl')
 rows=json.loads(source.read_text())['rows'] if old else [json.loads(l) for l in source.read_text().splitlines() if l.strip()]
 lookup={(str(r['id']),r['condition']):r for r in rows}; assert len(lookup)==len(rows)
 if a.action=='prepare':
  if review.exists(): raise SystemExit('Review already exists; not overwriting.')
  with review.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['id','condition','domain','question','rubric','unsupported_example','answer','limit']+FIELDS); w.writeheader()
   for r in rows:
    w.writerow(dict(id=r['id'],condition=r['condition'],domain=r.get('domain','11A.1'),question=r['question'],rubric=' | '.join(r['criteria']) if 'criteria' in r else r.get('expected',''),unsupported_example=r.get('unsupported_example',''),answer=r['answer'],limit=r.get('possibly_truncated')))
  print('Review:',review); return
 with review.open(newline='') as f: labels=list(csv.DictReader(f))
 if len(labels)!=len(rows) or {(r['id'],r['condition']) for r in labels}!=set(lookup): raise ValueError('Review rows do not match source')
 groups=defaultdict(list); candidates=[]; pending=0
 for r in labels:
  keys=['correct','complete','unsupported_count']+([] if old else FIELDS[:3]); vals=[r[k].strip() for k in keys]
  if not all(vals): pending+=1; continue
  if any(r[k] not in ('0','1') for k in keys if k!='unsupported_count'): raise ValueError(f'Invalid binary label: {r["id"]}')
  u=int(r['unsupported_count']); assert u>=0
  if u and not r['unsupported_quote'].strip(): raise ValueError('Provide unsupported claim quote: '+r['id'])
  correct=int(r['correct']); complete=int(r['complete']); criteria=None if old else sum(int(r[k]) for k in FIELDS[:3]); original=lookup[(r['id'],r['condition'])]
  if not old and correct and criteria!=3: raise ValueError('Correct requires all three criteria: '+r['id'])
  label='success' if correct and complete and not u else 'evidence_failure' if u and complete else 'other_failure'
  for key in [r['condition'],r['condition']+'/'+r['domain']]: groups[key].append((correct,complete,u,criteria,label))
  if not old and original['split']=='dev' and original.get('possibly_truncated') is False and label in ('success','evidence_failure'):
   candidates.append(dict(**original,review_label=label,unsupported_quote=r['unsupported_quote'],review_notes=r['notes']))
 summary={}
 for k,g in groups.items():
  n=len(g); summary[k]=dict(reviewed=n,accuracy_pct=100*sum(x[0] for x in g)/n,complete_pct=100*sum(x[1] for x in g)/n,unsupported_response_pct=100*sum(x[2]>0 for x in g)/n,unsupported_claims=sum(x[2] for x in g),mean_rubric_score=None if old else sum(x[3] for x in g)/n,evidence_failures=sum(x[4]=='evidence_failure' for x in g))
 result=dict(total=len(rows),pending=pending,groups=summary); (run/'review_metrics.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
 if not old and not pending and len(rows)==40 and all(r['split']=='dev' for r in rows):
  (run/'11C_candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in candidates)); print('11C candidate counts:',{k:sum(r['review_label']==k for r in candidates) for k in ('success','evidence_failure')})
 else: print('No new 11C export: requires fully reviewed 40-case development run.')
if __name__=='__main__': main()
