#!/usr/bin/env python3
"""Offline interactive reviewer for Hydra 11A.2 and 11A.1. Python 3.12+ stdlib."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

FIELDS = ['criterion_1','criterion_2','criterion_3','correct','complete','unsupported_count','unsupported_quote','failure_type','notes']
HEADER = ['id','condition','domain','question','rubric','unsupported_example','answer','limit'] + FIELDS

class Skip(Exception): pass
class Quit(Exception): pass

def ask(prompt, validate=lambda s: True):
    while True:
        s = input(prompt + ': ').strip()
        if s.lower() == ':skip': raise Skip()
        if s.lower() == ':quit': raise Quit()
        if validate(s): return s
        print('Invalid input. Try again (:skip or :quit).')

def binary(prompt): return ask(prompt + ' [0/1]', lambda s: s in ('0','1'))

def count_ok(s):
    try: return int(s) >= 0
    except ValueError: return False

def reviewed(r, old):
    keys = ['correct','complete','unsupported_count'] + ([] if old else FIELDS[:3])
    if not all(r[k].strip() for k in keys): return False
    if any(r[k] not in ('0','1') for k in keys if k != 'unsupported_count'): return False
    if not count_ok(r['unsupported_count']): return False
    if int(r['unsupported_count']) and not r['unsupported_quote'].strip(): return False
    if not old and r['correct']=='1' and any(r[k]!='1' for k in FIELDS[:3]): return False
    return True

def save(path, rows):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.review-', suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=HEADER); w.writeheader(); w.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)

def evaluate(r, original, old):
    print('\n'+'='*72)
    for key in ['id','condition','domain','question','answer','limit']:
        print(f'\n{key.upper()}:\n{r[key]}')
    print('\nRUBRIC:')
    if old: print(r['rubric'])
    else:
        for i,c in enumerate(original['criteria'],1): print(f'{i}. {c}')
    print('\nUNSUPPORTED EXAMPLE:',r['unsupported_example'] or '(none supplied)')
    print('A qualified hypothesis is not an unsupported assertion. Judge only the supplied evidence and technical facts.')
    d = dict(r)
    if not old:
        for i in range(1,4): d[f'criterion_{i}']=binary(f'Criterion {i} satisfied?')
    d['complete']=binary('Adequately addresses question?')
    d['unsupported_count']=str(int(ask('Distinct unsupported assertions [integer >= 0]',count_ok)))
    d['unsupported_quote']=ask('Exact unsupported statement(s), separated by |',lambda s: bool(s)) if int(d['unsupported_count']) else ''
    if (not old and any(d[k]=='0' for k in FIELDS[:3])) or int(d['unsupported_count']):
        d['correct']='0'; print('Strict correctness = 0 (missing criterion or unsupported claim).')
    else: d['correct']=binary('Technically correct with no material errors?')
    print('Failure types: none, unsupported_cause, factual_error, overconfidence, evidence_ignored, incorrect_diagnostic, fabrication, incomplete, other')
    d['failure_type']=ask('Failure type (free text)',lambda s: bool(s))
    d['notes']=ask('Review notes (optional)')
    if not reviewed(d,old): raise ValueError('Internal validation failed')
    return d

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path); p.add_argument('--revisit',action='store_true',help='Include already reviewed rows'); p.add_argument('--status',action='store_true',help='Show progress only'); a=p.parse_args()
    run=a.run.resolve(); path=run/'human_review_11A2.csv'; old=(run/'comparison.json').exists()
    source=run/('comparison.json' if old else 'responses.jsonl')
    originals=json.loads(source.read_text(encoding='utf-8'))['rows'] if old else [json.loads(l) for l in source.read_text(encoding='utf-8').splitlines() if l.strip()]
    lookup={(str(r['id']),r['condition']):r for r in originals}
    if len(lookup)!=len(originals): raise ValueError('Duplicate source IDs/conditions')
    if not path.exists():
        if a.status: print(f'0/{len(originals)} reviewed; CSV not prepared'); return
        scorer=Path(__file__).with_name('11A_2_human_review.py')
        subprocess.run([sys.executable,str(scorer),'prepare',str(run)],check=True)
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=HEADER: raise ValueError('CSV header differs from expected scoring schema; no changes made')
        rows=list(reader)
    keys=[(r['id'],r['condition']) for r in rows]
    if len(keys)!=len(lookup) or set(keys)!=set(lookup): raise ValueError('CSV rows do not match source')
    for r,k in zip(rows,keys):
        o=lookup[k]
        if r['question']!=o['question'] or r['answer']!=o['answer']: raise ValueError(f'CSV question/answer differs from source: {k}')
    def progress(): print(f'Reviewed: {sum(reviewed(r,old) for r in rows)}/{len(rows)}')
    progress()
    if a.status: return
    print('Close the CSV in other applications before reviewing. :skip skips current row; :quit exits without saving current unfinished row.')
    backup=path.with_name(path.name+'.'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.bak'); shutil.copy2(path,backup)
    print('Backup:',backup)
    try:
        for i,r in enumerate(rows):
            if reviewed(r,old) and not a.revisit: continue
            print(f'\nRow {i+1}/{len(rows)}')
            try: new=evaluate(r,lookup[keys[i]],old)
            except Skip: print('Skipped; previous values unchanged.'); continue
            rows[i]=new; save(path,rows); print('Saved.'); progress()
    except (Quit,KeyboardInterrupt,EOFError): print('\nStopped. Previously saved reviews retained.')
    progress(); print('Next: python Experiment11/11A_2_human_review.py score '+str(a.run))

if __name__=='__main__': main()
