"""Variance across independent complete edit searches, never pooled benchmark samples."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
from .spec import make_spec
from .audit_records import review_status

ROOT=Path(__file__).resolve().parents[3]

def aggregate(root=ROOT):
    groups=defaultdict(list);provisional=defaultdict(list);excluded=[];references={}
    for name in ("eval-base","eval-released-p50"):
        p=root/"replication/runs"/name/"heldout/summary.json"
        if p.exists():references[name]=json.loads(p.read_text())["sets"]
    for job in make_spec()["jobs"]:
        if job["experiment"]!=8:continue
        folder=root/"replication/runs"/job["id"]
        frozen=folder/"frozen.json";score=folder/"heldout/summary.json"
        if not frozen.exists() or not score.exists():continue
        meta=json.loads(frozen.read_text());sets=json.loads(score.read_text())["sets"]
        audit=review_status(folder)
        if audit not in ("passed","pending"):
            excluded.append({"job":job["id"],"semantic_protocol_audit":audit,
                "raw_scores_preserved":True})
            continue
        for benchmark,s in sets.items():
            if not s["complete"]:continue
            key=(job["model"],meta["transport"],job["passes"],benchmark)
            target=groups if audit=="passed" else provisional
            target[key].append({"repeat":job["repeat"],"job":job["id"],"accuracy":s["accuracy"],
                "cut_share":s["cut_share"],"semantic_protocol_audit":audit})
    return {"rows":group_rows(groups,references),"provisional_rows":group_rows(provisional,references),
        "excluded_runs":excluded,"references":references,
        "variance_unit":"independent search accuracy, ddof=1",
        "reference_replays_count_as_new_generations":False,"updated":time.time()}


def group_rows(groups,references):
    rows=[]
    for (model,transport,passes,benchmark),runs in sorted(groups.items()):
        assert len({r["repeat"] for r in runs})==len(runs),"Duplicate independent search"
        values=[r["accuracy"] for r in runs];base=references.get("eval-base",{}).get(benchmark,{})
        mean=statistics.mean(values)
        rows.append({"model":model,"transport":transport,"passes":passes,"benchmark":benchmark,
            "n_independent_searches":len(runs),"complete_three_repeats":len(runs)==3,"mean_accuracy":mean,
            "run_sample_variance":statistics.variance(values) if len(values)>1 else None,
            "run_sample_std":statistics.stdev(values) if len(values)>1 else None,
            "gain_vs_base":mean-base["accuracy"] if base.get("complete") else None,
            "audits":{r["job"]:r["semantic_protocol_audit"] for r in runs},"runs":runs})
    return rows

def main():
    out=ROOT/"replication/artifacts/math-summary.json";out.parent.mkdir(exist_ok=True)
    tmp=out.with_suffix(".tmp");tmp.write_text(json.dumps(aggregate(),indent=2));tmp.replace(out)

if __name__=="__main__":main()
