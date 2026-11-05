"""Trusted entry point inside the persistent, network/GPU-free editor namespace."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

torch.set_grad_enabled(False);torch.set_num_threads(4)
np.set_printoptions(precision=5,suppress=True,linewidth=150,threshold=100)
torch.set_printoptions(precision=5,threshold=100,linewidth=150)
BASE=Path("/model/model.safetensors");EDIT=Path("/work/edited");EDIT.mkdir(exist_ok=True)
with safe_open(str(BASE),framework="pt",device="cpu") as f:
    NAMES=[n for n in f.keys() if n.startswith("model.language_model.")]
FROZEN={"model.language_model.norm.weight","model.language_model.embed_tokens.weight","lm_head.weight"}


def load_original(name):
    if name=="lm_head.weight":name="model.language_model.embed_tokens.weight"
    if name not in NAMES:raise ValueError("Unknown tensor")
    with safe_open(str(BASE),framework="pt",device="cpu") as f:return f.get_tensor(name).float().clone()


def load(name):
    path=EDIT/(name+".npy")
    return torch.from_numpy(np.load(path,allow_pickle=False)).float() if path.exists() else load_original(name)


def save_edit(name,tensor):
    if name not in NAMES or name in FROZEN:raise ValueError("Tensor is out of scope")
    a=torch.as_tensor(tensor).detach().cpu().float();b=load_original(name)
    if a.shape!=b.shape or not torch.isfinite(a).all():raise ValueError("Invalid shape or nonfinite tensor")
    denom=(b.double()*b.double()).sum().item()
    if denom:
        scale=(a.double()*b.double()).sum().item()/denom
        error=torch.linalg.vector_norm((a-scale*b).flatten()).item()
        norm=max(torch.linalg.vector_norm(a.flatten()).item(),1e-12)
        if error/norm<1e-6 and not torch.equal(a,b):raise ValueError("Pure whole-tensor scaling is forbidden")
    path=EDIT/(name+".npy");tmp=path.with_suffix(".tmp.npy")
    np.save(tmp,a.numpy());tmp.replace(path)
    return {"name":name,"relative_change":float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))}


def drop_edit(name):
    (EDIT/(name+".npy")).unlink(missing_ok=True)


def edited():return sorted(p.name[:-4] for p in EDIT.glob("*.npy"))


tok=AutoTokenizer.from_pretrained("/model",local_files_only=True)
namespace={"np":np,"torch":torch,"NAMES":NAMES,"load":load,"load_original":load_original,
    "save_edit":save_edit,"drop_edit":drop_edit,"edited":edited,"tok":tok}


def check(code):
    tree=ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n,ast.Attribute) and n.attr in {"backward","autograd","optim","requires_grad_","set_grad_enabled"}:
            raise ValueError("Training/autograd operations are forbidden")
    return tree


for line in sys.stdin:
    try:
        request=json.loads(line);tree=check(request["code"]);out=io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(out):
            if tree.body and isinstance(tree.body[-1],ast.Expr):
                last=tree.body.pop();exec(compile(tree,"<editor>","exec"),namespace)
                value=eval(compile(ast.Expression(last.value),"<editor>","eval"),namespace)
                if value is not None:print(repr(value))
            else:exec(compile(tree,"<editor>","exec"),namespace)
        result={"ok":True,"out":out.getvalue()[:10000].encode("ascii","replace").decode()}
    except Exception as e:
        result={"ok":False,"out":(type(e).__name__+": "+str(e)+"\n"+traceback.format_exc()[-3000:])[:10000]}
    print(json.dumps(result),flush=True)
