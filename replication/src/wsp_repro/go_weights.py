"""The fixed 1,480,274-weight Go policy; no environment code in the editor."""
from pathlib import Path
import numpy as np

SHAPES={**{f"L{i}.W":(243 if i==1 else 512,512) for i in range(1,7)},
        **{f"L{i}.b":(512,) for i in range(1,7)},"out.W":(512,82),"out.b":(82,)}
NAMES=sorted(SHAPES)

def load_policy(folder):
    out={}
    for name,shape in SHAPES.items():
        path=Path(folder)/(name+".npy")
        a=np.load(path,allow_pickle=False) if path.exists() else np.zeros(shape,np.float32)
        if a.shape!=shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError("Invalid Go tensor: "+name)
        out[name]=a.astype(np.float32)
    return out

def forward(weights,obs,activations=False):
    x=np.asarray(obs,np.float32).reshape(-1,243);hidden=[]
    for i in range(1,7):
        x=np.maximum(x@weights[f"L{i}.W"]+weights[f"L{i}.b"],0)
        if activations:hidden.append(x.copy())
    logits=x@weights["out.W"]+weights["out.b"]
    if not np.isfinite(logits).all():raise ValueError("Nonfinite policy output")
    return (logits,hidden) if activations else logits
