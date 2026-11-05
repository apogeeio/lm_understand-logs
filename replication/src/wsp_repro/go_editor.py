"""Trusted prelude inside the CPU-only Go authoring namespace, without a Go library."""
import numpy as np
import torch
from pathlib import Path

torch.set_grad_enabled(False)
SHAPES={**{f"L{i}.W":(243 if i==1 else 512,512) for i in range(1,7)},
        **{f"L{i}.b":(512,) for i in range(1,7)},"out.W":(512,82),"out.b":(82,)}
NAMES=sorted(SHAPES);PASS=81
_folder=Path("/work/submission");_folder.mkdir(exist_ok=True)

def load(name):
    if name not in SHAPES:raise ValueError("Unknown tensor")
    p=_folder/(name+".npy")
    return np.load(p,allow_pickle=False).astype(np.float32) if p.exists() else np.zeros(SHAPES[name],np.float32)

def save(name,array):
    if name not in SHAPES:raise ValueError("Unknown tensor")
    a=np.asarray(array,np.float32)
    if a.shape!=SHAPES[name] or not np.isfinite(a).all():raise ValueError("Invalid shape or nonfinite tensor")
    p=_folder/(name+".npy");tmp=p.with_suffix(".tmp.npy");np.save(tmp,a);tmp.replace(p)

def saved():return sorted(n for n in NAMES if (_folder/(n+".npy")).exists())
def inp(row,col,plane):return (int(row)*9+int(col))*3+int(plane)
def mv(row,col):return int(row)*9+int(col)
