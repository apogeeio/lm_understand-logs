"""Check saved immutable text against the final math extractor without resampling."""
import hashlib
import json
import time
from .math_grading import extract,correct
from .run_math import ROOT

def main():
    gold={}
    for folder in (ROOT/"exp8_math/practice",ROOT/"replication/artifacts/bench"):
        for path in folder.glob("*.jsonl"):
            with path.open() as f:
                for row in map(json.loads,f):
                    if "answer" in row and not row.get("kind"):gold[row["id"]]=str(row["answer"])
    count=0;diff=[]
    paths=list((ROOT/"replication/runs").glob("*/sandbox/gen_*.jsonl"))+list((ROOT/"replication/runs").glob("*/heldout/eval-*.jsonl"))
    for path in paths:
        with path.open() as f:
            for row in map(json.loads,f):
                if row["id"] not in gold or row["correct"] is None:continue
                pred=extract(row["text"]);good=correct(pred,gold[row["id"]]);count+=1
                if good!=row["correct"] or pred!=row.get("pred",row.get("prediction")):
                    diff.append({"path":str(path.relative_to(ROOT)),"id":row["id"],"sample":row["sample"],
                        "before":row["correct"],"after":good,"prediction_changed":pred!=row.get("pred",row.get("prediction"))})
    from . import math_grading
    report={"rows":count,"differences":diff,"resampled":False,"updated":time.time(),
        "grader_sha256":hashlib.sha256(open(math_grading.__file__,"rb").read()).hexdigest()}
    (ROOT/"replication/artifacts/math-grading-audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=="__main__":main()
