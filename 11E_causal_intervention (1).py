import argparse, csv, hashlib, json, os, time, re
from pathlib import Path
import numpy as np
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
CONDITIONS=[('baseline',20,0),('zero_hook',20,0),('L20_025',20,.25),('L20_050',20,.5),('L20_100',20,1),('reverse',20,-.5),('random',20,.5),('L41_050',41,.5)]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)); tmp.replace(p)
def main():
    p=argparse.ArgumentParser(); p.add_argument('direction',type=Path); p.add_argument('--model',type=Path,default=Path('Models/gemma-4-e4b-mlx-4bit-transformers-default-v1')); p.add_argument('--dataset',type=Path,default=Path(__file__).with_name('hydra_11e_fresh.json')); p.add_argument('--smoke',action='store_true'); p.add_argument('--resume',type=Path); a=p.parse_args()
    import mlx.core as mx
    from mlx_vlm import load,generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    if not a.model.is_dir(): raise RuntimeError('Local model missing')
    pairs=json.loads(a.dataset.read_text()); assert len({r['id'] for r in pairs})==len(pairs)
    vectors={}
    for l in (20,41):
        with np.load(a.direction/f'candidate_L{l}.npz',allow_pickle=False) as z: v=z['direction'].astype(np.float32)
        if v.shape!=(2560,) or not np.isfinite(v).all() or abs(np.linalg.norm(v)-1)>1e-4: raise ValueError('Invalid direction')
        vectors[l]=v
    rng=np.random.default_rng(1127); r=rng.normal(size=2560).astype(np.float32); r-=vectors[20]*np.dot(r,vectors[20]); r/=np.linalg.norm(r)
    meta={'dataset':sha(a.dataset),'directions':{str(l):sha(a.direction/f'candidate_L{l}.npz') for l in vectors},'script':sha(__file__),'model':str(a.model.resolve()),'config':sha(a.model/'config.json'),'smoke':a.smoke,'conditions':CONDITIONS,'seed':1127,'max_tokens':96,'projection_validation':'dtype_aware_v2'}
    meta=json.loads(json.dumps(meta)); out=a.resume or Path('Experiment11/results')/('causal_'+str(time.time_ns())); out.mkdir(parents=True,exist_ok=bool(a.resume)); state=out/'checkpoint.json'
    if a.resume:
        saved=json.loads(state.read_text()); assert saved['meta']==meta,'Resume settings mismatch'; rows=saved['rows']
    else: rows=[]; atomic(state,{'meta':meta,'rows':rows})
    model,processor=load(str(a.model)); config=load_config(str(a.model)); lm=model.language_model; tok=getattr(processor,'tokenizer',processor)
    mods={l:lm.model.layers[l].mlp.down_proj for l in vectors}; classes={type(m) for m in mods.values()}; originals={c:c.__call__ for c in classes}; own={c:'__call__' in c.__dict__ for c in classes}
    for l,m in mods.items():
        if (m.bits,m.group_size,tuple(m.weight.shape))!=(8,64,(2560,2560)): raise RuntimeError(f'Unexpected quantization L{l}')
    def encode(s): return list(tok.encode(s,add_special_tokens=False))
    def prompt(q): return apply_chat_template(processor,config,q,num_images=0)
    def tokens(q,text):
        s=prompt(q); pre=encode(s); full=encode(s+text)
        if not pre or full[:len(pre)]!=pre or len(full)<=len(pre) or len(full)>256: raise RuntimeError('Unsafe token boundary or length')
        return pre,full
    active=None; stats=[]
    def wrap(original):
        def call(self,x,*args,**kwargs):
            y=original(self,x,*args,**kwargs)
            if active is None or self is not mods[active[1]]: return y
            name,l,alpha=active
            if x.shape[-1]!=10240 or y.shape[-1]!=2560: raise RuntimeError('Shape mismatch')
            if alpha==0: stats.append(0.); return y
            v=mx.array(r if name=='random' else vectors[l]); f=y.astype(mx.float32)
            proj=mx.sum(f*v,axis=-1,keepdims=True)
            ideal=f-alpha*proj*v
            z=ideal.astype(y.dtype)
            actual=z.astype(mx.float32)
            target=(1-alpha)*proj
            # Validate the FP32 operation separately from output-dtype rounding.
            # Scale arithmetic tolerance by the absolute dot-product terms,
            # not by a potentially near-zero signed projection.
            pre=mx.sum(ideal*v,axis=-1,keepdims=True)
            scale=mx.sum((mx.abs(f)+mx.abs(ideal))*mx.abs(v),axis=-1,keepdims=True)
            tol=2e-5*scale+1e-7
            pre_error=mx.abs(pre-target)
            # Round-to-nearest error bound for the actual output dtype.
            eps={mx.bfloat16:2**-7,mx.float16:2**-10,mx.float32:2**-23}.get(y.dtype)
            if eps is None: raise RuntimeError(f'Unsupported activation dtype: {y.dtype}')
            cast_bound=eps*mx.abs(ideal)+1e-37 if y.dtype!=mx.float16 else eps*mx.abs(ideal)+2**-24
            cast_error=mx.abs(actual-ideal)
            post_error=mx.abs(mx.sum(actual*v,axis=-1,keepdims=True)-target)
            post_bound=mx.sum(cast_bound*mx.abs(v),axis=-1,keepdims=True)+tol
            valid=mx.all(mx.isfinite(actual)) & mx.all(pre_error<=tol) & mx.all(cast_error<=cast_bound) & mx.all(post_error<=post_bound)
            # Retain the original relative residual as a diagnostic, not a gate.
            err=mx.sqrt(mx.sum(post_error**2))/(mx.sqrt(mx.sum(proj**2))+1e-6)
            mx.eval(valid,err); value=float(err.item())
            if not bool(valid.item()) or not np.isfinite(value):
                raise RuntimeError(f'Projection validation failed (dtype-aware): {value}, dtype={y.dtype}')
            stats.append(value); return z
        return call
    def score(q,text):
        pre,full=tokens(q,text); result=lm(mx.array([full],dtype=mx.int32)); logits=getattr(result,'logits',result)
        if logits.ndim!=3 or tuple(logits.shape[:2])!=(1,len(full)): raise RuntimeError('Logit alignment failed')
        z=logits[0,len(pre)-1:len(full)-1,:].astype(mx.float32); target=mx.array(full[len(pre):],dtype=mx.int32); lp=mx.take_along_axis(z,target[:,None],axis=-1).squeeze(-1)-mx.logsumexp(z,axis=-1); mx.eval(lp); mx.synchronize(); arr=np.array(lp)
        if not np.isfinite(arr).all(): raise RuntimeError('Nonfinite likelihood')
        return float(arr.mean()),float(arr.sum()),len(arr),arr
    def run_gen(q):
        mx.synchronize(); t=time.monotonic(); g=generate(model,processor,prompt(q),max_tokens=96,temperature=0.,verbose=False); mx.synchronize(); return g.text,int(g.generation_tokens),time.monotonic()-t
    first=pairs[0]; q=first['question']+' Answer in one sentence.'
    base=score(q,first['correct'])[3]; base_gen=run_gen(q)
    try:
        for c in classes: c.__call__=wrap(originals[c])
        active=CONDITIONS[1]; stats.clear(); zero=score(q,first['correct'])[3]; zero_gen=run_gen(q)
        if not stats or not np.allclose(base,zero,atol=1e-5,rtol=1e-5) or base_gen[:2]!=zero_gen[:2]: raise RuntimeError('Zero-hook equivalence failed')
        print('Zero-hook equivalence PASS',flush=True)
        for pair in (pairs[:1] if a.smoke else pairs):
            q=pair['question']+' Answer in one sentence.'
            for condition in CONDITIONS:
                name,l,alpha=condition
                if any(x['id']==pair['id'] and x['condition']==name for x in rows): continue
                active=None if name=='baseline' else condition; stats.clear()
                c,cs,cn,_=score(q,pair['correct']); w,ws,wn,_=score(q,pair['incorrect']); text,n,sec=run_gen(q)
                if active is not None and not stats: raise RuntimeError('Hook not called')
                row=dict(id=pair['id'],domain=pair['domain'],kind=pair['kind'],condition=name,layer=l,alpha=alpha,correct_mean=c,incorrect_mean=w,margin=c-w,total_margin=cs-ws,correct_tokens=cn,incorrect_tokens=wn,generated=text,generation_tokens=n,seconds=sec,possibly_truncated=n>=96,hook_calls=len(stats),projection_error=max(stats,default=0),expected=pair['correct'],incorrect=pair['incorrect'])
                rows.append(row); atomic(state,{'meta':meta,'rows':rows}); print(pair['id'],name,'margin',round(c-w,4),flush=True)
    finally:
        active=None
        for c in classes:
            if own[c]: c.__call__=originals[c]
            else: delattr(c,'__call__')
    restored=score(first['question']+' Answer in one sentence.',first['correct'])[3]; restored_gen=run_gen(first['question']+' Answer in one sentence.')
    if not np.allclose(base,restored,atol=1e-5,rtol=1e-5) or base_gen[:2]!=restored_gen[:2]: raise RuntimeError('Restoration failed')
    atomic(out/'validation.json',{'zero_hook_pass':True,'restoration_pass':True,'projection_validation':'dtype_aware_v2','completed':len(rows),'expected':len(CONDITIONS)*(1 if a.smoke else len(pairs))}); print('Restoration PASS\nOutput:',out)
if __name__=='__main__': main()
