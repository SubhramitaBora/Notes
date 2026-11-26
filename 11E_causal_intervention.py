"""Reusable MLX intervention, not a complete model hook."""
def remove_direction(output,direction,alpha=1.0):
    import mlx.core as mx
    u=direction/mx.maximum(mx.linalg.norm(direction),1e-8)
    return output-alpha*mx.sum(output*u,axis=-1,keepdims=True)*u
if __name__=='__main__':
    import mlx.core as mx
    y=mx.array([[2.,3.]]); u=mx.array([1.,0.]); result=remove_direction(y,u); mx.eval(result)
    assert result.tolist()==[[0.,3.]]
    print('Synthetic test passed. Integrate at verified activation site using local adapter; compare baseline, target and norm-matched control. No model edit performed.')
