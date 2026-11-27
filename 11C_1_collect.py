import argparse, json, time, inspect, hashlib
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='Models/gemma-4-e4b-mlx-4bit-transformers-default-v1')
    p.add_argument('--dataset',default=str(Path(__file__).with_name('hydra_contrast_pairs.json')))
    p.add_argument('--layers',default='0,20,41')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    path=Path(a.model)
    if not path.is_dir(): raise SystemExit('Local model directory missing')
    raw=Path(a.dataset).read_bytes(); pairs=json.loads(raw)
    assert len({r['id'] for r in pairs})==len(pairs)
    model,processor=load(str(path)); config=load_config(str(path))
    lm=model.language_model; layers=lm.model.layers
    tokenizer=getattr(processor,'tokenizer',processor)
    ids=sorted(set(map(int,a.layers.split(','))))
    assert ids and all(0<=i<len(layers) for i in ids)
    modules={i:layers[i].mlp.down_proj for i in ids}
    lookup={id(m):i for i,m in modules.items()}
    classes={type(m) for m in modules.values()}; originals={c:c.__call__ for c in classes}; own={c:'__call__' in c.__dict__ for c in classes}
    out=Path('Experiment11/results')/('contrast_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000000)); out.mkdir(parents=True)
    report={'pass':False,'dataset_sha256':hashlib.sha256(raw).hexdigest(),'layers':ids,'lm_signature':str(inspect.signature(lm.__call__)),'pairs':[],'scope':'Constructed continuations, not natural model failures; row t is state after consuming token t, predicts t+1.'}
    def save_report(): (out/'report.json').write_text(json.dumps(report,indent=2))
    save_report(); print('Language model signature:',report['lm_signature'],flush=True)
    captures={}; active=False; length=0; start=0
    def wrap(original):
        def call(self,x,*args,**kwargs):
            y=original(self,x,*args,**kwargs); i=lookup.get(id(self))
            if active and i is not None:
                if i in captures: raise RuntimeError('Repeated module call: alignment not established')
                if x.ndim!=3 or y.ndim!=3 or tuple(x.shape[:2])!=(1,length) or tuple(y.shape[:2])!=(1,length): raise RuntimeError(f'Layer {i}: unexpected sequence shapes {x.shape}, {y.shape}')
                xx=x[0,start:,:].astype(mx.float32); yy=y[0,start:,:].astype(mx.float32); mx.eval(xx,yy)
                captures[i]=(np.array(xx).copy(),np.array(yy).copy())
            return y
        return call
    def forward(tokens,begin,hook):
        nonlocal active,length,start
        captures.clear(); length=len(tokens); start=begin; active=hook
        try:
            result=lm(mx.array([tokens],dtype=mx.int32))
            logits=getattr(result,'logits',result)
            if logits.ndim!=3 or tuple(logits.shape[:2])!=(1,length): raise RuntimeError(f'Unexpected logits shape {logits.shape}')
            last=logits[0,-1,:].astype(mx.float32); mx.eval(last); mx.synchronize()
            return np.array(last).copy(),dict(captures)
        finally: active=False
    def encode(text): return list(tokenizer.encode(text,add_special_tokens=False))
    try:
        for c in classes: c.__call__=wrap(originals[c])
        for pair in (pairs[:1] if a.smoke else pairs):
            prompt=apply_chat_template(processor,config,pair['question']+' Answer in one sentence.',num_images=0)
            if not isinstance(prompt,str): raise RuntimeError('Expected rendered text prompt')
            prefix=encode(prompt); assert prefix
            entry={'id':pair['id'],'split':pair['split'],'branches':{}}
            branch_data={}
            for label in ('correct','incorrect'):
                text=pair[label]; full=encode(prompt+text)
                if full[:len(prefix)]!=prefix or len(full)<=len(prefix): raise RuntimeError('Tokenizer boundary changed; cannot align prefix safely')
                if len(full)>256: raise RuntimeError('Sequence exceeds safety cap')
                base,_=forward(full,len(prefix)-1,False)
                observed,data=forward(full,len(prefix)-1,True)
                if not np.isfinite(base).all() or not np.isfinite(observed).all() or not np.allclose(base,observed,rtol=1e-5,atol=1e-5): raise RuntimeError('Capture changed logits or nonfinite logits')
                if set(data)!=set(ids): raise RuntimeError('Missing layer capture')
                positions=np.arange(len(prefix)-1,len(full),dtype=np.int32)
                for i,(x,y) in data.items():
                    if x.shape!=(len(positions),10240) or y.shape!=(len(positions),2560) or not np.isfinite(x).all() or not np.isfinite(y).all(): raise RuntimeError(f'Layer {i} shape/finite validation failed')
                    np.savez_compressed(out/f"{pair['id']}_{label}_L{i}.npz",inputs=x,outputs=y,positions=positions,token_ids=np.array(full,dtype=np.int32)[positions])
                entry['branches'][label]={'continuation':text,'token_ids':full,'prefix_length':len(prefix),'captured_rows':len(positions),'max_logit_difference':float(np.max(np.abs(base-observed)))}
                branch_data[label]={i:(x[0].copy(),y[0].copy()) for i,(x,y) in data.items()}
            for i in ids:
                for x,y in zip(branch_data['correct'][i],branch_data['incorrect'][i]):
                    if not np.allclose(x,y,rtol=1e-4,atol=1e-4): raise RuntimeError(f'Common-prefix state differs at layer {i}')
            entry['common_prefix_check']=True; report['pairs'].append(entry); save_report(); print(pair['id'],'PASS',flush=True)
        report['pass']=True
    except Exception as e:
        report['error']=repr(e); raise
    finally:
        for c in classes:
            if own[c]: c.__call__=originals[c]
            else: delattr(c,'__call__')
        save_report(); print('Output:',out,flush=True)

if __name__=='__main__': main()
