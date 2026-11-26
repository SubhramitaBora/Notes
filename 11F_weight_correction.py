"""Dense reference math only. Never pass packed 4-bit weights."""
import numpy as np

def directional_weight_edit(W,direction,neuron_ids,alpha=1.0):
    W=np.asarray(W)
    if W.ndim!=2 or W.dtype.kind!='f': raise ValueError('Requires dense floating-point matrix, not packed weights')
    W=W.astype(np.float32); u=np.asarray(direction,dtype=np.float32); ids=np.unique(np.asarray(neuron_ids,dtype=int))
    if u.shape!=(W.shape[0],) or not np.isfinite(u).all() or np.linalg.norm(u)<1e-8: raise ValueError('Invalid direction')
    if not len(ids) or ids.min()<0 or ids.max()>=W.shape[1]: raise ValueError('Invalid neuron IDs')
    u=u/np.linalg.norm(u); patched=W.copy(); selected=W[:,ids]; patched[:,ids]=selected-alpha*np.outer(u,u@selected)
    return patched
if __name__=='__main__':
    W=np.array([[2.,4.],[3.,5.]],dtype=np.float32); result=directional_weight_edit(W,[1.,0.],[0]); assert np.allclose(result,[[0.,4.],[3.,5.]])
    print('Synthetic test passed. Quantization-aware Gemma integration pending causal evidence and architecture inspection. No checkpoint modified.')
