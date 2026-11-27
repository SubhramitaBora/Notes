"""Offline review and verified decision-replay export; no model required."""
import argparse,csv,json
from pathlib import Path
from collections import Counter

def main():
 p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','score']); p.add_argument('run',type=Path); a=p.parse_args(); out=a.run
 state=json.loads((out/'checkpoint.json').read_text()); rows=state['rows']; expected=1 if state['meta']['smoke'] else 50
 if len(rows)!=expected:raise ValueError('Complete inference before review')
 path=out/'semantic_review.csv'; lookup={r['id']:r for r in rows}; fields=['id','domain','split','question','options','expected','criterion','answer','decision_status','margin','semantic_correct','complete','unsupported_count','unsupported_quote','decision_verified','notes']
 if a.action=='prepare':
  with path.open('x',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
   for r in rows:w.writerow(dict(id=r['id'],domain=r['domain'],split=r['split'],question=r['question'],options=json.dumps(r['options']),expected=r['correct'],criterion=r['criteria'][0],answer=r['answer'],decision_status=r['decision_status'],margin=r['margin']))
  print('Review:',path);return
 with path.open(newline='') as f: labels=list(csv.DictReader(f))
 if len(labels)!=len(rows) or set(r['id'] for r in labels)!=set(lookup):raise ValueError('Review IDs mismatch')
 counts=Counter(); exports=[]
 for l in labels:
  r=lookup[l['id']]; keys=['semantic_correct','complete','unsupported_count','decision_verified']
  if any(not l[k].strip() for k in keys):counts['pending']+=1;continue
  if any(l[k] not in ('0','1') for k in keys if k!='unsupported_count'):raise ValueError('Invalid binary label '+r['id'])
  u=int(l['unsupported_count']); quote=l['unsupported_quote'].strip()
  if u<0 or (u and (not quote or quote not in r['answer'])):raise ValueError('Unsupported quote must occur verbatim in answer '+r['id'])
  if l['answer']!=r['answer'] or l['expected']!=r['correct']:raise ValueError('Source columns altered')
  counts['reviewed']+=1; counts['semantic_correct']+=int(l['semantic_correct']); counts['complete']+=int(l['complete']); counts['unsupported_responses']+=int(u>0); counts['unsupported_claims']+=u
  if l['decision_verified']=='1' and r['decision_status'] in ('wrong','correct') and not r['possibly_truncated']:
   exports.append(dict(id=r['id'],split=r['split'],domain=r['domain'],label='decision_failure' if r['decision_status']=='wrong' else 'decision_success',prefix_token_ids=r['prefix_token_ids'],position=len(r['prefix_token_ids'])-1,correct_next_token_id=r['correct_token_id'],incorrect_next_token_id=r['incorrect_token_id'],observed_decision=r['decision'],margin=r['margin'],formatted_prompt=r['formatted_prompt'],answer=r['answer'],semantic_correct=int(l['semantic_correct']),unsupported_quote=quote,scope='Verified text decision; teacher-forced replay, not actual generation token trace'))
 summary=dict(counts); (out/'semantic_report.txt').write_text(json.dumps(summary,indent=2)+'\n'); print(summary)
 if counts['pending']:print('Export blocked until all reviews complete');return
 if state['meta']['smoke']:print('Smoke: no research export');return
 (out/'11E2_verified_replay.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in exports)); print('Replay counts:',dict(Counter((r['split']+' '+r['label']) for r in exports)))
if __name__=='__main__':main()
