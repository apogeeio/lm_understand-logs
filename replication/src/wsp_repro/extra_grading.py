"""Official IFEval strict checkers and multiple-choice extraction after thinking."""
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]

def mcq_extract(text):
    text=text.split("</think>")[-1].replace("**","")
    match=re.search(r"answer is \(?([A-J])\)?",text)
    if match:return match.group(1)
    match=re.search(r".*[aA]nswer:\s*([A-J])",text)
    if match:return match.group(1)
    match=re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)",text,re.DOTALL)
    return match.group(0) if match else None

def ifeval(row,text):
    vendor=ROOT/"replication/vendor/ifeval"
    if str(vendor) not in sys.path:sys.path.insert(0,str(vendor))
    import nltk
    import langdetect
    nltk.data.path.insert(0,str(vendor/"nltk_data")) if str(vendor/"nltk_data") not in nltk.data.path else None
    langdetect.DetectorFactory.seed=0
    from instruction_following_eval import instructions_registry as registry
    answer=text.split("</think>")[-1].strip() if "</think>" in text else ""
    follows=[]
    for iid,kw in zip(row["instruction_id_list"],row["kwargs"]):
        ins=registry.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**{k:v for k,v in (kw or {}).items() if v is not None})
        args=ins.get_instruction_args()
        if args and "prompt" in args:ins.build_description(prompt=row["prompt"])
        follows.append(bool(answer) and bool(ins.check_following(answer)))
    return all(follows),None,{"inst_ok":sum(follows),"inst_total":len(follows)}
