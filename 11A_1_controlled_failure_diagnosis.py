"""Hydra 11A.1: local, inference-only controlled diagnosis.
Run from any directory. Original 11A schema and generation API retained.
"""
import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / 'Models' / 'gemma-4-e4b-mlx-4bit-transformers-default-v1'
BASELINE = ROOT / 'results' / 'failures_20261126_112643.jsonl'
INSTRUCTION = ('\n\nAnswer in at most 100 words. State what the evidence establishes, '
               'what remains unknown, and the most useful next checks. '
               'Avoid unsupported claims and unnecessary introductions.')

def repetition_score(text, n=8):
    words = re.findall(r'\S+', text.lower())
    grams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return 1-len(set(grams))/len(grams) if grams else 0.0

def read_baseline(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 12 or len({r['id'] for r in rows}) != 12:
        raise ValueError('Expected 12 unique baseline cases.')
    for r in rows:
        for key in ('id','question','expected','answer','seconds','generation_tokens','possibly_truncated','repetition_score'):
            if key not in r:
                raise ValueError(f'Missing baseline field: {key}')
        if not isinstance(r['question'], str) or not isinstance(r['answer'], str):
            raise ValueError('Question and answer must be strings.')
    return rows

def record(r, condition, budget, prompt, answer, tokens, seconds, flag, finish=None):
    return dict(id=r['id'], condition=condition, question=r['question'], expected=r['expected'],
                prompt=prompt, answer=answer, max_tokens=budget, generation_tokens=tokens,
                possibly_truncated=flag, finish_reason=finish, seconds=seconds,
                repetition_score=repetition_score(answer), words=len(answer.split()),
                human_correct_0_or_1=None, human_complete_0_or_1=None,
                human_unsupported_claim_count=None, human_failure_type=None, human_notes=None)

def avg(rows, key):
    values = [r[key] for r in rows if isinstance(r[key], (int,float)) and not isinstance(r[key], bool)]
    return mean(values) if values else None

def save_reports(out, rows):
    with (out/'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {}
    for condition in ('baseline','extended','concise'):
        subset = [r for r in rows if r['condition']==condition]
        if not subset: continue
        known = [r for r in subset if isinstance(r['possibly_truncated'], bool)]
        hit = sum(r['possibly_truncated'] is True for r in known)
        summary[condition] = dict(n=len(subset), token_limit_hits=hit,
            unknown_limit_status=len(subset)-len(known), token_limit_rate_pct=100*hit/len(known) if known else None,
            mean_tokens=avg(subset,'generation_tokens'), mean_words=avg(subset,'words'),
            mean_seconds=avg(subset,'seconds'), mean_repetition=avg(subset,'repetition_score'))
    baseline = {r['id']:r for r in rows if r['condition']=='baseline'}
    pairs = []
    for r in rows:
        if r['condition']=='baseline': continue
        b=baseline[r['id']]
        pairs.append(dict(id=r['id'], condition=r['condition'], baseline_limit=b['possibly_truncated'],
            modified_limit=r['possibly_truncated'], word_change=r['words']-b['words'],
            seconds_change=r['seconds']-b['seconds'],
            token_change=r['generation_tokens']-b['generation_tokens'] if isinstance(r['generation_tokens'],int) and isinstance(b['generation_tokens'],int) else None))
    (out/'comparison.json').write_text(json.dumps(dict(summary=summary, paired_comparisons=pairs, rows=rows), indent=2), encoding='utf-8')
    lines=['HYDRA 11A.1 — CONTROLLED FAILURE DIAGNOSIS', 'Token-limit hits are a proxy for truncation, not accuracy or completeness.','Human review fields are intentionally blank.']
    for name,s in summary.items():
        lines.append(f'\n{name.upper()}')
        lines.extend(f'  {k}: {v:.4f}' if isinstance(v,float) else f'  {k}: {v}' for k,v in s.items())
    (out/'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--model',type=Path,default=MODEL)
    args=parser.parse_args()
    source=read_baseline(args.baseline)
    if not args.model.is_dir(): raise FileNotFoundError(args.model)
    import mlx.core as mx
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    out=ROOT/'results'/f'controlled_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True)
    rows=[record(r,'baseline',256,r['question'],r['answer'],r['generation_tokens'],r['seconds'],r['possibly_truncated'],r.get('finish_reason')) for r in source]
    (out/'settings.json').write_text(json.dumps(dict(model=str(args.model),baseline=str(args.baseline),temperature=0.0,extended_budget=1024,concise_budget=256,concise_instruction=INSTRUCTION,order='paired extended then concise',weight_edits=False),indent=2))
    save_reports(out,rows)
    print('Loading local model...',flush=True)
    model,processor=load(str(args.model)); config=load_config(str(args.model))
    print('Results:',out,flush=True)
    try:
        for i,r in enumerate(source,1):
            for condition,budget,suffix in [('extended',1024,''),('concise',256,INSTRUCTION)]:
                q=r['question']+suffix
                prompt=apply_chat_template(processor,config,q,num_images=0)
                mx.synchronize(); start=time.perf_counter()
                result=generate(model,processor,prompt,max_tokens=budget,temperature=0.0,verbose=False)
                mx.synchronize(); seconds=time.perf_counter()-start
                answer=getattr(result,'text',str(result)); tokens=getattr(result,'generation_tokens',None)
                tokens=int(tokens) if tokens is not None else None
                finish=getattr(result,'finish_reason',None)
                row=record(r,condition,budget,q,answer,tokens,seconds,tokens>=budget if tokens is not None else None,str(finish) if finish is not None else None)
                rows.append(row); save_reports(out,rows)
                print(f'{i}/12 {condition}: tokens={tokens}, limit={row["possibly_truncated"]}, seconds={seconds:.3f}',flush=True)
                del result
    finally:
        print(save_reports(out,rows)); print('\nSaved:',out)
        print(f'New generations completed: {len(rows)-12}/24')

if __name__=='__main__':
    main()
