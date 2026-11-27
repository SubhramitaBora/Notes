import argparse,csv,hashlib,inspect,json,os,random,re,time
from pathlib import Path
from importlib.metadata import version
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parent
CONDS=['P0','P1','P2','P3','P4']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 t=p.with_suffix('.tmp'); t.write_text(json.dumps(x,indent=2,allow_nan=False)); t.replace(p)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,default=ROOT.parent/'Models/gemma-4-e4b-mlx-4bit-transformers-default-v1'); p.add_argument('--smoke',action='store_true'); p.add_argument('--resume',type=Path); a=p.parse_args()
 data=ROOT/'hydra_11p_20.json'; cases=json.loads(data.read_text()); assert len(cases)==20 and len({c['id'] for c in cases})==20
 if a.smoke: cases=cases[:1]
 import mlx.core as mx
 from mlx.utils import tree_flatten,tree_unflatten
 from mlx_vlm import load
 try:
  from mlx_vlm import stream_generate
 except ImportError:
  from mlx_vlm.generate.dispatch import stream_generate
 from mlx_vlm.prompt_utils import apply_chat_template
 from mlx_vlm.utils import load_config
 # Only use an explicitly exposed prefill chunk-size option; never silently substitute.
 import importlib
 ar=importlib.import_module('mlx_vlm.generate.ar'); step=getattr(ar,'generate_step',None)
 if step is None or 'prefill_step_size' not in inspect.signature(step).parameters: raise RuntimeError('P4 unsupported: inspect local generate_step signature; no benchmark started')
 param=inspect.signature(step).parameters['prefill_step_size']; default=param.default
 if not isinstance(default,int) or default<=0: raise RuntimeError('Unknown prefill default')
 optimized=max(1,default//2)
 if optimized==default: raise RuntimeError('No distinct prefill setting')
 model,processor=load(str(a.model)); config=load_config(str(a.model)); name='language_model.model.layers.0.self_attn.q_norm.weight'; original=dict(tree_flatten(model.parameters()))[name]; flat=original.reshape(-1)
 if 'float' not in str(original.dtype): raise RuntimeError('Patch target not floating')
 edited=mx.where(mx.arange(flat.size)==0,flat.astype(mx.float32)-.02,flat.astype(mx.float32)).astype(original.dtype).reshape(original.shape); mx.eval(edited)
 assert int(mx.sum(edited!=original).item())==1
 actual=float(edited.reshape(-1)[0].item())-float(flat[0].item())
 def update(v): model.update(tree_unflatten([(name,v)])); mx.eval(v)
 def restored(): return bool(mx.all(dict(tree_flatten(model.parameters()))[name]==original).item())
 def infer(c,k):
  q=c['question']+'\nA. '+c['A']+'\nB. '+c['B']+'\nStart with A or B, then explain your reasoning.'
  if k in ('P1','P3'): q+=' Limit explanation to 60 words; include evidence and any uncertainty.'
  prompt=apply_chat_template(processor,config,q,num_images=0); kw=dict(max_tokens=256,temperature=0.0)
  if k=='P4': kw['prefill_step_size']=optimized
  try:
   if k in ('P2','P3'): update(edited)
   mx.synchronize(); mx.reset_peak_memory(); start=time.perf_counter(); first=None; chunks=[]; last=None
   for event in stream_generate(model,processor,prompt,**kw):
    mx.synchronize(); last=event; s=event.text; chunks.append(s)
    if s and first is None: first=time.perf_counter()-start
   mx.synchronize(); elapsed=time.perf_counter()-start; peak=mx.get_peak_memory()/2**20
   if last is None: raise RuntimeError('Empty stream')
   text=''.join(chunks); n=int(last.generation_tokens); m=re.match(r'^\s*([AB])(?=\s|[.:)]|$)',text); choice=m.group(1) if m else ''
   return dict(answer=text,choice=choice,correct=int(choice==c['correct']),seconds=elapsed,ttft=first,tokens=n,tps=n/elapsed,peak_mib=peak,cap_hit=int(n>=256))
  finally:
   update(original)
   if not restored(): raise RuntimeError('Restoration failed')
 meta=dict(script=sha(__file__),dataset=sha(data),model=str(a.model.resolve()),config=sha(a.model/'config.json'),mlx=version('mlx'),mlx_vlm=version('mlx-vlm'),smoke=a.smoke,patch=name,actual_delta=actual,prefill_default=default,prefill_P4=optimized,seed=42,max_tokens=256)
 out=a.resume or ROOT/'results'/f'performance_{time.time_ns()}'; out.mkdir(parents=True,exist_ok=bool(a.resume)); state=out/'checkpoint.json'; rows=[]
 if a.resume:
  saved=json.loads(state.read_text()); assert saved['meta']==meta,'Resume mismatch'; rows=saved['rows']
 else: atomic(state,dict(meta=meta,rows=rows))
 rng=random.Random(42); jobs=[]
 for rep in range(1,2 if a.smoke else 4):
  order=cases.copy(); rng.shuffle(order)
  for c in order:
   ks=CONDS.copy(); rng.shuffle(ks); jobs.extend((rep,c,k) for k in ks)
 done={(r['rep'],r['id'],r['condition']) for r in rows}; assert len(done)==len(rows) and done<={(rep,c['id'],k) for rep,c,k in jobs}
 print('Output:',out,flush=True); print('Patch actual delta:',actual,'P4 prefill:',default,'->',optimized,flush=True)
 warm=dict(question='A service returns HTTP 200. What does that establish?',A='Request succeeded',B='All dependencies are healthy',correct='A')
 for k in CONDS: infer(warm,k)
 for rep,c,k in jobs:
  if (rep,c['id'],k) in done: continue
  r=infer(c,k); rows.append(dict(rep=rep,id=c['id'],domain=c['domain'],condition=k,expected=c['correct'],**r)); atomic(state,dict(meta=meta,rows=rows)); print(f'{len(rows)}/{len(jobs)} {c["id"]} {k} {r["seconds"]:.3f}s correct={r["correct"]}',flush=True)
 print('Restoration PASS; completed',len(rows),'/',len(jobs)); print('Output:',out)
if __name__=='__main__': main()
