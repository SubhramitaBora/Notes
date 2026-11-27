"""Local baseline decision discovery. No hooks or parameter edits."""
import argparse,csv,hashlib,json,os,re,time,math
from pathlib import Path
from collections import Counter
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
 t=p.with_suffix('.tmp'); t.write_text(json.dumps(obj,indent=2,allow_nan=False)); t.replace(p)
def csvout(p,rows):
 if not rows:return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def parse(text):
 m=re.match(r'^\s*([AB])(?=\s|[.:)]|$)',text)
 return m.group(1) if m else None

def report(out,rows,expected):
 csvout(out/'responses.csv',rows)
 lines=[f'Completed {len(rows)}/{expected}', 'Margins: correct minus incorrect mean token log likelihood; nats. Low = 0 to 0.25 inclusive.']
 for group in ['all','discovery','check']:
  rr=[r for r in rows if group=='all' or r['split']==group]; n=len(rr)
  if not n:continue
  counts=Counter(r['decision_status'] for r in rr)
  lines.append(f'{group}: n={n}; decisions={dict(counts)}; negative={sum(r["margin"]<0 for r in rr)}; low={sum(0<=r["margin"]<=.25 for r in rr)}; mean margin={sum(r["margin"] for r in rr)/n:.4f}; cap hits={sum(r["possibly_truncated"] for r in rr)}')
 lines.append('Semantic accuracy not scored. Check split is preliminary, not final validation.')
 (out/'report.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,default=ROOT.parent/'Models/gemma-4-e4b-mlx-4bit-transformers-default-v1'); p.add_argument('--dataset',type=Path,default=ROOT/'hydra_11e1_50.json'); p.add_argument('--smoke',action='store_true'); p.add_argument('--resume',type=Path); a=p.parse_args()
 cases=json.loads(a.dataset.read_text()); assert len(cases)==50 and len({r['id'] for r in cases})==50
 assert Counter(r['domain'] for r in cases)==dict.fromkeys(['Kubernetes','Docker','Jenkins','Linux','AWS'],10)
 assert Counter(r['correct'] for r in cases)=={'A':25,'B':25}
 if a.smoke: cases=cases[:1]
 if not a.model.is_dir(): p.error('Local model missing')
 meta=dict(dataset=sha(a.dataset),script=sha(__file__),model=str(a.model.resolve()),config=sha(a.model/'config.json'),smoke=a.smoke,max_tokens=192,temperature=0)
 out=a.resume or ROOT/'results'/f'genuine_{time.time_ns()}'; out.mkdir(parents=True,exist_ok=bool(a.resume)); state=out/'checkpoint.json'
 if a.resume:
  saved=json.loads(state.read_text()); assert saved['meta']==meta,'Resume settings mismatch'; rows=saved['rows']
 else: rows=[]; atomic(state,dict(meta=meta,rows=rows))
 assert len({r['id'] for r in rows})==len(rows) and set(r['id'] for r in rows)<=set(r['id'] for r in cases)
 print('Output:',out,flush=True)
 import mlx.core as mx
 import numpy as np
 from mlx_vlm import load,generate
 from mlx_vlm.prompt_utils import apply_chat_template
 from mlx_vlm.utils import load_config
 model,processor=load(str(a.model)); config=load_config(str(a.model)); lm=model.language_model; tok=getattr(processor,'tokenizer',processor)
 def encode(s):return list(tok.encode(s,add_special_tokens=False))
 def score(pre,full):
  if not pre or full[:len(pre)]!=pre or len(full)<=len(pre) or len(full)>512:raise RuntimeError('Unsafe token alignment or length')
  result=lm(mx.array([full],dtype=mx.int32)); logits=getattr(result,'logits',result)
  if logits.ndim!=3 or tuple(logits.shape[:2])!=(1,len(full)):raise RuntimeError('Logit shape mismatch')
  z=logits[0,len(pre)-1:len(full)-1,:].astype(mx.float32); target=mx.array(full[len(pre):],dtype=mx.int32)
  lp=mx.take_along_axis(z,target[:,None],axis=-1).squeeze(-1)-mx.logsumexp(z,axis=-1); mx.eval(lp); arr=np.array(lp); assert np.isfinite(arr).all(); return float(arr.mean())
 for c in cases:
  if any(r['id']==c['id'] for r in rows):continue
  q=c['question']+'\nWhich action or conclusion is best?\nA. '+c['options']['A']+'\nB. '+c['options']['B']+'\nStart your answer with A or B, then explain your reasoning in at most 80 words.'
  prompt=apply_chat_template(processor,config,q,num_images=0); assert isinstance(prompt,str); pre=encode(prompt)
  # Identical label-only continuations; balance label assignment across domains.
  branches={k:encode(prompt+k) for k in ('A','B')}
  if any(len(v)!=len(pre)+1 or v[:len(pre)]!=pre for v in branches.values()) or branches['A'][-1]==branches['B'][-1]:raise RuntimeError('Expected distinct single-token labels; inspect tokenizer before proceeding')
  scores={k:score(pre,v) for k,v in branches.items()}; margin=scores[c['correct']]-scores[c['incorrect']]
  mx.synchronize(); t=time.monotonic(); g=generate(model,processor,prompt,max_tokens=192,temperature=0.,verbose=False); mx.synchronize(); elapsed=time.monotonic()-t
  text=g.text; n=int(g.generation_tokens); decision=parse(text); status='unparsed' if decision is None else 'correct' if decision==c['correct'] else 'wrong'
  # Text re-encoding is not guaranteed to reproduce actual sampled token IDs.
  # Only export a verified replay prefix later; never claim actual token trace.
  row=dict(**c,condition='baseline',formatted_prompt=prompt,prefix_token_ids=pre,correct_token_id=branches[c['correct']][-1],incorrect_token_id=branches[c['incorrect']][-1],correct_mean=scores[c['correct']],incorrect_mean=scores[c['incorrect']],margin=margin,answer=text,decision=decision,decision_status=status,generation_tokens=n,possibly_truncated=n>=192,seconds=elapsed)
  rows.append(row); atomic(state,dict(meta=meta,rows=rows)); print(f'{len(rows)}/{len(cases)} {c["id"]} {status} margin={margin:.4f}',flush=True); del g
 report(out,rows,len(cases))
if __name__=='__main__':main()
