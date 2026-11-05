"""Durable generation queue: never resend an already-recorded successful response."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .spec import make_spec
from .prompts import make_prompt
from .edits import apply
from .code_policy import extract
from .providers import call,MissingCredential

ROOT=Path(__file__).resolve().parents[2]


def generate(job):
    out=ROOT/"runs"/job["id"];out.mkdir(parents=True,exist_ok=True)
    if (out/"artifact.json").exists():return job["id"],"already_frozen"
    state=out/"state.json"
    if state.exists():
        previous=json.loads(state.read_text())
        retryable={"RateLimited","TransientProviderError","URLError","TimeoutError","QuotaExhausted"}
        if previous.get("status")=="failed" and previous.get("error_type") not in retryable:
            return job["id"],"retained_failure"
    prompt=make_prompt(job["game"],job["kind"],job.get("width",32))
    (out/"job.json").write_text(json.dumps(job,indent=2))
    (out/"prompt.txt").write_text(prompt)
    try:
        # An API answer saved before a crash is reused, never regenerated to improve its score.
        if (out/"call/answer.md").exists() and (out/"call/call.json").exists():
            text=(out/"call/answer.md").read_text();metadata=json.loads((out/"call/call.json").read_text())
        else:text,metadata=call(job["model"],job["effort"],[{"role":"user","content":prompt}],out/"call")
        if job["kind"]=="network":
            tensors,report=apply(text,width=job["width"])
            artifact=out/"network.npz";np.savez(artifact,**tensors)
        else:
            code,report=extract(text,job["max_code_lines"])
            artifact=out/"policy.py";artifact.write_text(code)
        record={"job":job,"parse":report,"transport":metadata["transport"],
                "artifact":artifact.name,"sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "frozen_before_evaluation":True,"created":time.time()}
        (out/"artifact.json").write_text(json.dumps(record,indent=2))
        return job["id"],"frozen"
    except MissingCredential:
        (out/"state.json").write_text(json.dumps({"status":"waiting_credentials"}))
        return job["id"],"waiting_credentials"
    except Exception as e:
        (out/"state.json").write_text(json.dumps({"status":"failed","error_type":type(e).__name__,"message":str(e)}))
        return job["id"],"failed"


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--model",choices=["astra","deepseek"])
    ap.add_argument("--limit",type=int)
    ap.add_argument("--evaluate",action="store_true")
    args=ap.parse_args()
    jobs=[j for j in make_spec()["jobs"] if j["experiment"]==5 and (not args.model or j["model"]==args.model)]
    if args.limit:jobs=jobs[:args.limit]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job,result in zip(jobs,pool.map(generate,jobs)):
            print(json.dumps(result),flush=True)
            if args.evaluate and result[1] in ("frozen","already_frozen"):
                from .dispatch_eval import evaluate_frozen
                evaluate_frozen(job)


if __name__=="__main__":main()
