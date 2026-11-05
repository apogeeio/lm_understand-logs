import numpy as np
from wsp_repro.go_weights import SHAPES,load_policy,forward

def test_go_zero_network_and_direct_pass_bias(tmp_path):
    assert sum(np.prod(s) for s in SHAPES.values())==1480274
    w=load_policy(tmp_path)
    assert all(a.dtype==np.float32 for a in w.values())
    x=np.zeros((2,243),np.float32)
    assert np.array_equal(forward(w,x),np.zeros((2,82)))
    w["out.b"][81]=4
    lg,act=forward(w,x,True)
    assert np.all(lg.argmax(1)==81) and len(act)==6
