"""Validate the short-code baseline before running it only in a Linux sandbox."""
import ast
import re

IMPORTS={"numpy","math","collections","heapq"}
DENIED={"open","exec","eval","compile","__import__","globals","locals","vars","getattr","setattr",
        "delattr","input","breakpoint","memoryview","type","help","dir"}
DENIED_ATTR={"load","save","savez","savez_compressed","loadtxt","savetxt","fromfile","tofile","memmap",
             "random","ctypeslib","set_printoptions","seterr","get_include"}


def extract(text,max_lines=40):
    blocks=re.findall(r"```python\s*\n(.*?)```",text,re.S)
    if not blocks:raise ValueError("No complete Python block")
    code=blocks[-1]
    count=sum(bool(l.strip()) and not l.lstrip().startswith("#") for l in code.splitlines())
    if count>max_lines:raise ValueError(f"{count} lines exceeds {max_lines}")
    tree=ast.parse(code)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(funcs)!=1 or funcs[0].name!="act":raise ValueError("Define exactly one top-level function act(obs)")
    f=funcs[0]
    if len(f.args.args)!=1 or f.args.defaults or f.args.kwonlyargs or f.args.vararg or f.args.kwarg or f.decorator_list:
        raise ValueError("Use plain act(obs), no defaults/decorators")
    if any(not isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)) for n in tree.body):
        raise ValueError("Top level may only import and define act")
    for n in ast.walk(tree):
        if isinstance(n,(ast.Global,ast.Nonlocal,ast.AsyncFunctionDef,ast.ClassDef)):
            raise ValueError("Persistent/global/class state is forbidden")
        if isinstance(n,ast.Name) and (n.id in DENIED or n.id.startswith("__")):
            raise ValueError("Forbidden name")
        if isinstance(n,ast.Attribute) and (n.attr.startswith("_") or n.attr in DENIED_ATTR):
            raise ValueError("Forbidden attribute")
        if isinstance(n,ast.Import) and any(x.name not in IMPORTS for x in n.names):
            raise ValueError("Import not allowed")
        if isinstance(n,ast.ImportFrom) and (n.module not in IMPORTS or n.level):
            raise ValueError("Import not allowed")
        if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            if any(isinstance(t,ast.Attribute) for t in targets):raise ValueError("Attribute assignment forbidden")
    return code,{"physical_code_lines":count,"static_checks_passed":True,"requires_linux_sandbox":True}
