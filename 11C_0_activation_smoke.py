import argparse, json, time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Models/gemma-4-e4b-mlx-4bit-transformers-default-v1')
    p.add_argument('--layers', default='0,20,41')
    p.add_argument('--max-tokens', type=int, default=64)
    args = p.parse_args()
    path = Path(args.model)
    if not path.is_dir(): raise SystemExit(f'Local model not found: {path}')
    model, processor = load(str(path))
    config = load_config(str(path))
    layers = model.language_model.model.layers
    ids = sorted(set(map(int, args.layers.split(','))))
    if any(i < 0 or i >= len(layers) for i in ids): raise ValueError('Invalid layer index')
    modules = {i: layers[i].mlp.down_proj for i in ids}
    lookup = {id(m): i for i,m in modules.items()}
    classes = {type(m) for m in modules.values()}
    originals = {c: c.__call__ for c in classes}
    own = {c: '__call__' in c.__dict__ for c in classes}
    captures = {}
    question = 'Pod lastState terminated reason OOMKilled, exit 137; node reports no MemoryPressure. What is established and how distinguish container limit from node OOM? Answer briefly.'
    prompt = apply_chat_template(processor, config, question, num_images=0)
    def run():
        mx.synchronize()
        r = generate(model, processor, prompt, max_tokens=args.max_tokens, temperature=0.0, verbose=False)
        mx.synchronize()
        return r.text, r.generation_tokens
    baseline, bt = run()
    def wrap(original):
        def call(self, x, *a, **kw):
            y = original(self, x, *a, **kw)
            i = lookup.get(id(self))
            if i is not None:
                records = captures.setdefault(i, [])
                # Keep only last position per call, not full sequence or graph.
                if len(records) < 128:
                    xx = x.reshape(-1, x.shape[-1])[-1].astype(mx.float32)
                    yy = y.reshape(-1, y.shape[-1])[-1].astype(mx.float32)
                    mx.eval(xx, yy)
                    records.append((np.array(xx).copy(), np.array(yy).copy(), list(x.shape), list(y.shape)))
            return y
        return call
    try:
        for c in classes: c.__call__ = wrap(originals[c])
        observed, ot = run()
    finally:
        for c in classes:
            if own[c]: c.__call__ = originals[c]
            else: delattr(c, '__call__')
    out = Path('Experiment11/results') / ('activation_smoke_' + time.strftime('%Y%m%d_%H%M%S') + '_' + str(time.time_ns()%1000000000))
    out.mkdir(parents=True, exist_ok=False)
    report = {'question':question,'baseline':baseline,'captured_response':observed,'baseline_tokens':bt,'captured_tokens':ot,'text_equal':baseline==observed,'layers':{},'scope':'Last position per module call; not aligned to generated token IDs; no quality labels.'}
    for i in ids:
        rows = captures.get(i, [])
        if not rows:
            report['layers'][str(i)] = {'calls_captured':0}; continue
        x = np.stack([r[0] for r in rows]); y = np.stack([r[1] for r in rows])
        np.savez_compressed(out/f'layer_{i}.npz', inputs=x, outputs=y)
        report['layers'][str(i)] = {'calls_captured':len(rows),'input_shape':list(x.shape),'output_shape':list(y.shape),'first_call_input_shape':rows[0][2],'first_call_output_shape':rows[0][3],'finite':bool(np.isfinite(x).all() and np.isfinite(y).all()),'bits':getattr(modules[i],'bits',None),'group_size':getattr(modules[i],'group_size',None)}
    report['pass'] = bool(baseline==observed and bt==ot and all(r.get('finite',False) for r in report['layers'].values()))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2)); print('Output:',out)
    if not report['pass']: raise SystemExit('Smoke test failed; inspect report before proceeding.')

if __name__ == '__main__': main()
