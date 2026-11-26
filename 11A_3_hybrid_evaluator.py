"""Evaluate saved development answers. New output only; resume by input fingerprint."""
import argparse,json,hashlib,os,sys
from pathlib import Path
from hydra_rules import inspect,normalize
ROOT=Path(__file__).resolve().parent

def read_jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)); tmp.replace(p)
def inputs(run,bench):
    cases=[x for x in json.loads(bench.read_text()) if x['split']=='dev']; rows=read_jsonl(run/'responses.jsonl'); lookup={x['id']:x for x in cases}
    if len(cases)!=40 or len(lookup)!=40 or len(rows)!=40 or {r['id'] for r in rows}!=set(lookup): raise ValueError('Expected exactly 40 unique development responses')
    for r in rows:
        for k in ('question','criteria','domain','split','unsupported_example'):
            if r[k]!=lookup[r['id']][k]: raise ValueError(f'Mismatch {r["id"]}: {k}')
        if r['condition']!='baseline' or not isinstance(r['answer'],str) or len(r['criteria'])!=3: raise ValueError('Invalid response')
    return rows

def combine(row,passes):
    rules=inspect(row); flags=[]; scores=[p.get('score') for p in passes]
    if any(s is None for s in scores): flags.append('invalid_judge_output')
    if rules['limit'] is not False: flags.append('answer_limit_or_unknown')
    if rules['example_literal_match']: flags.append('literal_unsupported_example_needs_context')
    score=None
    if all(s is not None for s in scores):
        a,b=scores
        keys=('criteria','complete','failure_type','correct','unsupported_count')
        if any(a[k]!=b[k] for k in keys) or set(map(normalize,a['unsupported_quotes']))!=set(map(normalize,b['unsupported_quotes'])): flags.append('judge_disagreement')
        if a['uncertain'] or b['uncertain']: flags.append('judge_uncertain')
        if any(a['criteria'][i] and not c['hits'] for i,c in enumerate(rules['lexical_coverage'])): flags.append('semantic_credit_without_lexical_support')
        if not flags: score=a
    label=None
    if score:
        if score['correct'] and score['complete']: label='success'
        elif score['unsupported_count'] and score['complete']: label='evidence_failure'
    return {'id':row['id'],'domain':row['domain'],'rules':rules,'passes':passes,'flags':flags,'score':score,'candidate_label':label}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run',type=Path); p.add_argument('--benchmark',type=Path,default=ROOT/'hydra_evidence_60.json'); p.add_argument('--model',type=Path,default=ROOT.parent/'Models/gemma-4-e4b-mlx-4bit-transformers-default-v1'); p.add_argument('--smoke',action='store_true'); p.add_argument('--rules-only',action='store_true'); a=p.parse_args()
    rows=inputs(a.run,a.benchmark)
    if a.smoke: rows=rows[:2]
    if not a.rules_only and not a.model.is_dir(): p.error('Local model directory missing')
    out=a.run/('auto11a3_rules' if a.rules_only else 'auto11a3_smoke' if a.smoke else 'auto11a3_full'); out.mkdir(exist_ok=True)
    files=[a.run/'responses.jsonl',a.benchmark,Path(__file__),ROOT/'hydra_rules.py',ROOT/'hydra_judge.py']
    meta={'hashes':{str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},'model':str(a.model.resolve()),'smoke':a.smoke,'rules_only':a.rules_only,'count':len(rows)}
    if not a.rules_only: meta['model_files']={str(x.relative_to(a.model)):[x.stat().st_size,x.stat().st_mtime_ns] for x in sorted(a.model.rglob('*')) if x.is_file()}
    settings=out/'settings.json'
    if settings.exists() and json.loads(settings.read_text())!=meta: raise SystemExit('Inputs changed; preserve/rename output folder before new run')
    write(settings,meta)
    judge=None
    for i,r in enumerate(rows,1):
        dest=out/(r['id']+'.json')
        if dest.exists(): print(f'{i}/{len(rows)} {r["id"]} resumed'); continue
        if a.rules_only: record={'id':r['id'],'domain':r['domain'],'rules':inspect(r),'flags':['not_semantically_graded'],'score':None,'candidate_label':None}
        else:
            if judge is None:
                from hydra_judge import Judge
                judge=Judge(a.model)
            record=combine(r,[judge.evaluate(r,0),judge.evaluate(r,1)])
        write(dest,record); print(f'{i}/{len(rows)} {r["id"]}: {record["flags"] or record["candidate_label"] or "other_failure"}',flush=True)
    print('Output:',out)
if __name__=='__main__': main()
