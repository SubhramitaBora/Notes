import csv,json,sys,statistics
from pathlib import Path
p=Path(sys.argv[1]); rows=list(csv.DictReader((p/'semantic_review.csv').open())); source=json.loads((p/'checkpoint.json').read_text())['rows']; keys=lambda r:(r['id'],int(r['rep']),r['condition']); original={keys(r):r for r in source}
assert len(rows)==len(original) and {keys(r) for r in rows}==set(original)
for r in rows:
 assert r['answer']==original[keys(r)]['answer'],'Answer modified'
pending=sum(any(r[k]=='' for k in ['semantic_correct','complete','unsupported_count']) for r in rows)
if pending: print('Pending:',pending); sys.exit(1)
for r in rows:
 assert r['semantic_correct'] in ('0','1') and r['complete'] in ('0','1') and int(r['unsupported_count'])>=0
lines=['Human semantic review; reviewer judgments, not automated accuracy.']
for k in ['P0','P1','P2','P3','P4']:
 rr=[r for r in rows if r['condition']==k]; lines.append(f'{k}: semantic accuracy {100*statistics.mean(int(r["semantic_correct"]) for r in rr):.2f}%; completeness {100*statistics.mean(int(r["complete"]) for r in rr):.2f}%; unsupported response rate {100*statistics.mean(int(r["unsupported_count"])>0 for r in rr):.2f}%')
(p/'semantic_report.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
