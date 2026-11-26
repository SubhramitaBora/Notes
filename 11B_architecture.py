import json
from pathlib import Path
from mlx_vlm import load
from mlx.utils import tree_flatten
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    model,processor=load(str(ROOT.parent/'Models'/'gemma-4-e4b-mlx-4bit-transformers-default-v1'))
    params=tree_flatten(model.parameters()); records=[]
    for name,tensor in params:
        if 'mlp' in name or 'q_norm' in name:
            records.append(dict(name=name,shape=list(tensor.shape),dtype=str(tensor.dtype)))
            if 'down_proj' in name: print(records[-1])
    out=ROOT/'results'; out.mkdir(exist_ok=True); (out/'architecture.json').write_text(json.dumps(records,indent=2)); print('Total tensors:',len(params),'Relevant:',len(records))
