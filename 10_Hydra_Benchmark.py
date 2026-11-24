import argparse, json, csv, time, random, re, hashlib
from pathlib import Path
from datetime import datetime
from importlib.metadata import version

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'benchmarks/hydra_100.json'
DEFAULT=ROOT/'Models/gemma-4-e4b-mlx-4bit-transformers-default-v1'
CANDIDATES={'baseline':None,'L0_minus':(0,-.02),'L0_plus':(0,.02),'L1_minus':(1,-.02),'L1_plus':(1,.02)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,default=DEFAULT); p.add_argument('--stage',choices=['smoke','dev','test'],required=True); p.add_argument('--winner',choices=list(CANDIDATES)[1:]); a=p.parse_args()
    if a.stage=='test' and not a.winner: p.error('Supply the development winner with --winner')
    if not a.model.is_dir(): p.error(f'Model missing: {a.model}')
    import mlx.core as mx
    from mlx.utils import tree_flatten,tree_unflatten
    from mlx_vlm import load,generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    cases=json.loads(DATA.read_text()); cases=[c for c in cases if c['split']==('test' if a.stage=='test' else 'dev')]
    if a.stage=='smoke': cases=cases[:2]
    variants=['baseline',a.winner] if a.stage=='test' else list(CANDIDATES)
    model,processor=load(str(a.model)); config=load_config(str(a.model)); params=dict(tree_flatten(model.parameters()))
    originals={}; edits={}; details={}
    for v in variants[1:]:
        layer,delta=CANDIDATES[v]; name=f'language_model.model.layers.{layer}.self_attn.q_norm.weight'; original=params[name]
        if 'float' not in str(original.dtype): raise ValueError(f'Not floating: {name}')
        originals[name]=original; flat=original.reshape(-1); edited=mx.where(mx.arange(flat.size)==0,flat.astype(mx.float32)+delta,flat.astype(mx.float32)).astype(original.dtype).reshape(original.shape); mx.eval(edited)
        count=int(mx.sum(edited!=original).item()); actual=float(edited.reshape(-1)[0].item())-float(flat[0].item())
        if count!=1: raise ValueError(f'{v}: expected exactly one changed value, got {count}')
        edits[v]=(name,edited); details[v]=dict(parameter=name,requested_delta=delta,actual_delta=actual,changed_scalars=count)
    del params
    def update(name,value):
        model.update(tree_unflatten([(name,value)])); mx.eval(value)
    def infer(question):
        prompt=apply_chat_template(processor,config,question,num_images=0); start=time.perf_counter(); result=generate(model=model,processor=processor,prompt=prompt,max_tokens=128,temperature=0.0,verbose=False); elapsed=time.perf_counter()-start
        text=getattr(result,'text',str(result)); tokens=getattr(result,'generation_tokens',getattr(result,'output_tokens',None)); finish=getattr(result,'finish_reason',None)
        match=re.fullmatch(r'\s*([ABCD])\s*[.)]?\s*',text,re.I); choice=match.group(1).upper() if match else ''
        return dict(response=text,choice=choice,seconds=round(elapsed,6),tokens=tokens,finish_reason=finish,possibly_truncated=finish=='length' or (tokens is not None and tokens>=128))
    print('Warming up...'); infer('Reply with the single letter A.')
    out=ROOT/'evaluation_results'/'hydra10'/f'{a.stage}_{datetime.now():%Y%m%d_%H%M%S}'; out.mkdir(parents=True)
    (out/'settings.json').write_text(json.dumps(dict(stage=a.stage,variants=variants,case_ids=[c['id'] for c in cases],dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),model=str(a.model),mlx_vlm=version('mlx-vlm'),mlx=version('mlx'),max_tokens=128,temperature=0,patches=details),indent=2))
    rng=random.Random(42); total=len(cases)*len(variants); n=0
    with (out/'results.csv').open('w',newline='') as f:
        writer=None
        for c in cases:
            order=variants.copy(); rng.shuffle(order)
            question='Choose the single best answer. Reply with ONLY A, B, C or D.\n'+c['question']+'\n'+'\n'.join(f'{k}. {v}' for k,v in c['options'].items())
            for v in order:
                try:
                    if v!='baseline': update(*edits[v])
                    result=infer(question)
                finally:
                    if v!='baseline': update(edits[v][0],originals[edits[v][0]])
                row=dict(case_id=c['id'],category=c['category'],variant=v,**result)
                if writer is None: writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader()
                writer.writerow(row); f.flush(); n+=1; print(f'{n}/{total} {c["id"]} {v}: {result["choice"] or "INVALID"} {result["seconds"]:.2f}s',flush=True)
    print('Saved:',out)
if __name__=='__main__': main()
