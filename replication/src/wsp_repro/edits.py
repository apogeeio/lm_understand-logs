"""Read numeric edits with AST, never eval/exec model text; retain per-statement errors."""
import ast
import re
import numpy as np

KEYS = {"W1":"actor/dense1/kernel","b1":"actor/dense1/bias",
        "W2":"actor/dense2/kernel","b2":"actor/dense2/bias",
        "W3":"actor/logits/kernel","b3":"actor/logits/bias"}


def zeros(width=32):
    return {"W1":np.zeros((600,width),np.float32),"b1":np.zeros(width,np.float32),
            "W2":np.zeros((width,width),np.float32),"b2":np.zeros(width,np.float32),
            "W3":np.zeros((width,6),np.float32),"b3":np.zeros(6,np.float32)}


def literal(node):
    value=ast.literal_eval(node)
    a=np.asarray(value)
    if a.dtype.kind not in "iuf" or not np.isfinite(a).all() or np.any(np.abs(a)>1e6):
        raise ValueError("Only finite decimal numbers of magnitude <= 1e6 are allowed")
    return value


def index(node):
    if isinstance(node,ast.Tuple):return tuple(index(x) for x in node.elts)
    if isinstance(node,ast.Slice):
        return slice(*(index(x) if x is not None else None for x in [node.lower,node.upper,node.step]))
    value=ast.literal_eval(node)
    if type(value) is not int:raise ValueError("Index must be an integer")
    return value


def apply(text, tensors=None, width=32):
    tensors=zeros(width) if tensors is None else {k:v.copy() for k,v in tensors.items()}
    blocks=re.findall(r"```edits\s*\n(.*?)```",text,re.S)
    report={"block":bool(blocks),"applied":0,"errors":[]}
    if not blocks:return tensors,report
    source=re.sub(r"(?m)^([ \t]*)ZERO\s+([A-Za-z_][\w.]*)\s*(?:#.*)?$",r"\1\2[:] = 0",blocks[-1])
    try:body=ast.parse(source).body
    except SyntaxError as e:
        report["errors"].append(f"syntax line {e.lineno}: {e.msg}");return tensors,report
    for node in body:
        try:
            if isinstance(node,ast.Assign) and len(node.targets)==1:target=node.targets[0];op=None
            elif isinstance(node,ast.AugAssign):target=node.target;op=node.op
            else:raise ValueError("Only tensor assignments are allowed")
            if not isinstance(target,ast.Subscript):
                raise ValueError("Target must be a named tensor slice")
            if isinstance(target.value,ast.Name):name=target.value.id
            elif isinstance(target.value,ast.Attribute) and isinstance(target.value.value,ast.Name):
                name=target.value.value.id+"."+target.value.attr
            else:raise ValueError("Target must be a named tensor slice")
            if name not in tensors:raise ValueError("Unknown tensor")
            idx=index(target.slice);value=literal(node.value)
            # Work on a copy, so a failed statement cannot leave a partial update.
            a=tensors[name].copy()
            if op is None:a[idx]=value
            elif isinstance(op,ast.Add):a[idx]+=value
            elif isinstance(op,ast.Sub):a[idx]-=value
            elif isinstance(op,ast.Mult):a[idx]*=value
            else:raise ValueError("Unsupported update")
            if not np.isfinite(a).all():raise ValueError("Non-finite result")
            tensors[name]=a;report["applied"]+=1
        except (ValueError,TypeError,IndexError,OverflowError) as e:
            report["errors"].append(f"line {node.lineno}: {type(e).__name__}: {e}")
    report["statements"]=len(body)
    report["nonzero"]=sum(int(np.count_nonzero(a)) for a in tensors.values())
    return tensors,report


def logits(tensors, obs):
    x=np.asarray(obs,np.float32).reshape(600)
    h=np.tanh(x@tensors["W1"]+tensors["b1"])
    h=np.tanh(h@tensors["W2"]+tensors["b2"])
    return h@tensors["W3"]+tensors["b3"]
