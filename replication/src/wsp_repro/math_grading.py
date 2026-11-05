"""Boxed-answer extraction and math-verify grading, independent of private weightenv."""

def extract(text):
    text=text.split("</think>")[-1]
    start=text.rfind("\\boxed")
    if start<0:start=text.rfind("\\fbox")
    if start<0:return None
    boxed=text.startswith("\\boxed",start)
    rest=text[start+(6 if boxed else 5):]
    s=rest.lstrip()
    if not s.startswith("{"):
        out=rest.strip().split("$")[0].split("\n")[0].strip() if boxed and rest.startswith(" ") else None
        return out or None
    depth=0
    for i,c in enumerate(s):
        depth+=(c=="{")-(c=="}")
        if depth==0:return s[1:i].strip() or None
    return None

def correct(pred,gold):
    from math_verify import parse,verify
    if pred is None:return False
    try:
        g,p=parse("\\boxed{"+gold+"}",parsing_timeout=5),parse("\\boxed{"+pred+"}",parsing_timeout=5)
        return bool(g and p and verify(g,p,timeout_seconds=5))
    except Exception:return False
