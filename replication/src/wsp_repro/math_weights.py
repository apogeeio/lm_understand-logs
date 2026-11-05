"""Independent enforcement of the original editor's allowed weight scope."""
import json
from pathlib import Path
import numpy as np

FROZEN={"model.language_model.norm.weight","model.language_model.embed_tokens.weight","lm_head.weight"}


def names(base):
    from safetensors import safe_open
    with safe_open(Path(base)/"model.safetensors",framework="pt",device="cpu") as f:
        return [n for n in f.keys() if n.startswith("model.language_model.")]


def original(base,name):
    from safetensors import safe_open
    with safe_open(Path(base)/"model.safetensors",framework="pt",device="cpu") as f:
        return f.get_tensor(name).float()


def validate(base,edit_dir):
    import torch
    allowed=set(names(base));out=[]
    for path in sorted(Path(edit_dir).glob("*.npy")):
        name=path.name[:-4]
        if name not in allowed or name in FROZEN:raise ValueError("Out-of-scope tensor: "+name)
        a=torch.from_numpy(np.load(path,allow_pickle=False)).float();b=original(base,name)
        if a.shape!=b.shape or not torch.isfinite(a).all():raise ValueError("Invalid shape/nonfinite: "+name)
        # Reject a pure whole-tensor rescaling, including a pure sign flip/zeroing.
        denom=(b.double()*b.double()).sum().item()
        if denom:
            scale=(a.double()*b.double()).sum().item()/denom
            residual=torch.linalg.vector_norm((a-scale*b).reshape(-1)).item()
            norm=max(torch.linalg.vector_norm(a.reshape(-1)).item(),1e-12)
            if residual/norm<1e-6 and not torch.equal(a,b):
                raise ValueError("Whole-tensor rescaling is forbidden: "+name)
        out.append(name)
    return out
