"""Validate official strict checkers against all released first-sample grades."""
import gzip
import json
from .extra_grading import ifeval,ROOT

def main():
    with (ROOT/"replication/artifacts/bench/ifeval.jsonl").open() as f:
        rows={r["id"]:r for r in map(json.loads,f)}
    total=0;diff=[]
    for arm in ("unedited","p10c","p25c","p50c"):
        for path in (ROOT/"exp8_math/eval"/arm/"parts").glob("ifeval*/*_first_samples.jsonl.gz"):
            truth={r["id"]:bool(r["correct"][0]) for r in json.loads(path.with_name("ifeval_per_problem.json").read_text())}
            with gzip.open(path,"rt") as f:
                for item in map(json.loads,f):
                    good,_,_=ifeval(rows[item["id"]],item["text"]);total+=1
                    if good!=truth[item["id"]]:diff.append({"arm":arm,"id":item["id"],"expected":truth[item["id"]],"actual":good})
    report={"checked":total,"mismatches":diff,"language_detector_seed":0,
        "checker_source_commit":"e6890f85757dd84e27ca6df2dd30651dafad28e0"}
    (ROOT/"replication/artifacts/ifeval-reference-grading.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=="__main__":main()
