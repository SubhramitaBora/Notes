"""Local Evidence-60 inference; defaults to development only."""
import argparse,json,time,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--stage',choices=['dev','test'],default='dev'); p.add_argument('--confirm-held-out',action='store_true'); p.add_argument('--smoke',action='store_true'); p.add_argument('--model',type=Path,default=ROOT.parent/'Models/gemma-4-e4b-mlx-4bit-transformers-default-v1'); a=p.parse_args()
 if a.stage=='test' and not a.confirm_held_out: p.error('Reserve test for frozen final validation; explicit confirmation required.')
 source=ROOT/'hydra_evidence_60.json'; cases=[r for r in json.loads(source.read_text()) if r['split']==a.stage]; assert len(cases)==(40 if a.stage=='dev' else 20)
 if a.smoke: cases=cases[:1]
 if not a.model.is_dir(): raise FileNotFoundError(a.model)
 import mlx.core as mx
 from mlx_vlm import load,generate
 from mlx_vlm.prompt_utils import apply_chat_template
 from mlx_vlm.utils import load_config
 out=ROOT/'results'/f'evidence_{a.stage}_{datetime.now():%Y%m%d_%H%M%S_%f}'; out.mkdir(parents=True)
 instruction='\n\nIn at most 100 words, state what the evidence establishes, what remains unknown, and the most useful next checks. Avoid unsupported claims.'
 (out/'settings.json').write_text(json.dumps(dict(stage=a.stage,smoke=a.smoke,model=str(a.model),temperature=0,max_tokens=512,instruction=instruction,benchmark_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),expected_count=len(cases),weight_edits=False),indent=2))
 model,processor=load(str(a.model)); config=load_config(str(a.model))
 print('Results:',out,flush=True)
 with (out/'responses.jsonl').open('w') as f:
  for i,c in enumerate(cases,1):
   q=c['question']+instruction; prompt=apply_chat_template(processor,config,q,num_images=0)
   mx.synchronize(); t=time.perf_counter(); result=generate(model,processor,prompt,max_tokens=512,temperature=0.0,verbose=False); mx.synchronize(); elapsed=time.perf_counter()-t
   tokens=getattr(result,'generation_tokens',None); tokens=int(tokens) if tokens is not None else None
   r=dict(**c,condition='baseline',prompt=q,formatted_prompt=prompt,answer=getattr(result,'text',str(result)),generation_tokens=tokens,possibly_truncated=tokens>=512 if tokens is not None else None,seconds=elapsed,finish_reason=str(getattr(result,'finish_reason',None)))
   f.write(json.dumps(r,default=str)+'\n'); f.flush(); print(f'{i}/{len(cases)} {c["id"]}: tokens={tokens}, seconds={elapsed:.3f}',flush=True); del result
 print('Completed:',out)
if __name__=='__main__': main()
