"""Regrade immutable replies after fixing Markdown section extraction; no new calls."""
import hashlib
import json
import time
from .run_reading import grade,ROOT
from .audit_records import valid_review

def main():
    count=0;changed=0
    for path in (ROOT/"replication/runs").glob("e[123]-*/result.json"):
        job=json.loads(path.with_name("job.json").read_text())
        turns=sorted(path.parent.glob("turn-*/answer.md"))
        if not turns:continue
        reply=turns[-1];old=json.loads(path.read_text());new=grade(job,reply.read_text())
        new.update(transport=old["transport"],turns=old["turns"])
        review=valid_review(path.parent)
        if review:
            new["semantic_label_review"]=review.get("label_review","pending")
            if job["experiment"]==3:new["semantic_no_forward_pass_audit"]=review["verdict"]
            if review.get("descriptive_detail_review"):
                new["descriptive_detail_review"]=review["descriptive_detail_review"]
        if old!=new:
            receipt=path.with_name("grade-correction-label-section-v2.json")
            if not receipt.exists():receipt.write_text(json.dumps({"before":old,"after":new,
                "reason":"Recognize Markdown labels and headings without reading reasoning as the chosen answer",
                "answer_sha256":hashlib.sha256(reply.read_bytes()).hexdigest(),"resampled":False,
                "updated":time.time()},indent=2))
            path.write_text(json.dumps(new,indent=2));changed+=old.get("correct")!=new["correct"]
        count+=1
    print(json.dumps({"graded":count,"correctness_changes":changed,"model_calls":0}))

if __name__=="__main__":main()
