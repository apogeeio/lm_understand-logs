"""Low-priority, resumable held-out chunks after immutable edit submission."""
from .execution_scope import math_paused
import hashlib
import json
from pathlib import Path
import shutil
import time
from .spec import make_spec

ROOT=Path(__file__).resolve().parents[3]
QUEUE=ROOT/"replication/artifacts/math-forward-queue"
SAMPLES={"math500":4,"gsm8k":2,"amc23":16,"minerva":4,"olympiad":4,"ifeval":2,"mmlupro1k":1,"gpqa":4}

def freeze_evaluation(job):
    folder=ROOT/"replication/runs"/job["id"];frozen=folder/"frozen.json"
    if job["id"]!="eval-base" and not frozen.exists():return None
    work=folder/"heldout";work.mkdir(parents=True,exist_ok=True);edit=work/"edited";edit.mkdir(exist_ok=True)
    expected=json.loads(frozen.read_text())["edited"] if frozen.exists() else {}
    for name,digest in expected.items():
        src=folder/"sandbox/edited"/(name+".npy");dst=edit/src.name
        assert hashlib.sha256(src.read_bytes()).hexdigest()==digest,"Frozen submission changed"
        if not dst.exists():shutil.copyfile(src,dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest,"Evaluation edit differs from frozen submission"
    assert {p.name[:-4] for p in edit.glob("*.npy")}==set(expected)
    (work/"submission.json").write_text(json.dumps({"job":job["id"],"edited":expected,
        "frozen_before_heldout":True},indent=2))
    return work

def advance(job,manifest):
    if math_paused(ROOT):return
    work=freeze_evaluation(job)
    if work is None:return
    scope_path=ROOT/"replication/configs/benchmark-deferrals.json"
    deferred=json.loads(scope_path.read_text()) if scope_path.exists() else {}
    deferred={name:record for name,record in deferred.items()
        if name in SAMPLES and record["status"]=="deferred"}
    summary={};pending=False
    # Complete a deferred set's accounting without submitting further chunks.
    # Remaining sets start with the lowest total generation count; token lengths vary.
    order=sorted(SAMPLES,key=lambda name:(name not in deferred,
        manifest.get(name,{}).get("count",0)*SAMPLES[name]))
    for name in order:
        n=SAMPLES[name]
        if name not in manifest:continue
        chunks=[];total=manifest[name]["count"];size=16//n
        for start in range(0,total,size):
            end=min(start+size,total);uid=f"eval-{job['id']}-{name}-{start:04d}-{end:04d}"
            response=QUEUE/(uid+".response.json")
            if response.exists():
                result=json.loads(response.read_text())
                if not result["ok"]:
                    # Keep the original error and request; no automatic retries alter output evidence.
                    (work/(uid+".error.json")).write_text(json.dumps(result));return
                assert result["result"]["data_sha256"]==manifest[name]["sha256"],"Held-out data changed"
                chunks.append(result["result"]);continue
            if name in deferred:break
            path=QUEUE/(uid+".request.json");processing=path.with_suffix(".processing")
            if not path.exists() and not processing.exists():
                envelope={"work":str(work),"pass_id":uid,"request":{"type":"heldout","benchmark":name,
                    "start":start,"end":end}}
                tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(envelope));tmp.replace(path)
            pending=True;break # At most one pending chunk per checkpoint.
        if name in deferred and not chunks:continue
        gens=sum(c["generations"] for c in chunks)
        summary[name]={"completed_problems":sum(c["end"]-c["start"] for c in chunks),
            "expected_problems":total,"complete":gens==total*n,"samples":n,"generations":gens,
            "accuracy":sum(c["ok"] for c in chunks)/gens if gens else None,
            "cut_share":sum(c["cut"] for c in chunks)/gens if gens else None,
            "tokens":sum(c["tokens"] for c in chunks),"seconds":sum(c["seconds"] for c in chunks)}
        if name in deferred:
            summary[name]["deferred"]=True
        if pending:break
    # Other benchmark adapters are deliberately marked pending, not silently excluded.
    report={"job":job,"protocol":"think35","max_new_tokens":32768,"sets":summary,
        "pending_adapters":[],"pending_data":[s for s in SAMPLES if s not in manifest],
        "deferred_benchmarks":deferred,
        "current_phase_benchmarks_complete":all(name in summary and summary[name]["complete"]
            for name in SAMPLES if name not in deferred),
        "all_requested_benchmarks_complete":len(summary)==len(SAMPLES) and all(s["complete"] for s in summary.values()),
        "protocol_notes":{"mmlupro1k":manifest.get("mmlupro1k",{}).get("private_original_prompt_helper"),
            "gpqa":manifest.get("gpqa",{}).get("original_private_prompt")},
        "updated":time.time()}
    (work/"summary.json").write_text(json.dumps(report,indent=2))

def main():
    if math_paused(ROOT):return
    QUEUE.mkdir(parents=True,exist_ok=True)
    jobs=[{"id":"eval-base","model":"base"},{"id":"eval-released-p50","model":"released_reference","calibration_only":True}]+[j for j in make_spec()["jobs"] if j["experiment"]==8]
    while True:
        path=ROOT/"replication/artifacts/bench/manifest.json"
        if path.exists():
            manifest=json.loads(path.read_text())
            for job in jobs:advance(job,manifest)
            from .aggregate_math import main as aggregate_math
            aggregate_math()
        time.sleep(30)

if __name__=="__main__":main()
