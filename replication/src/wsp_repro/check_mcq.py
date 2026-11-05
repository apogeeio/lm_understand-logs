"""Compare official MMLU-Pro extraction with released first-sample outcomes."""
import gzip
import json
from .prepare_bench import ROOT,BENCH
from .extra_grading import mcq_extract

def main():
    with (BENCH/"mmlupro1k.jsonl").open() as f:rows={r["id"]:r for r in map(json.loads,f)}
    total=0;diff=[]
    for arm in ("unedited","p10c","p25c","p50c"):
        for path in (ROOT/"exp8_math/eval"/arm/"parts").glob("mmlupro1k*/*_first_samples.jsonl.gz"):
            truth={r["id"]:bool(r["correct"][0]) for r in json.loads(path.with_name("mmlupro1k_per_problem.json").read_text())}
            with gzip.open(path,"rt") as f:
                for item in map(json.loads,f):
                    pred=mcq_extract(item["text"]);good=pred==rows[item["id"]]["answer"];total+=1
                    if good!=truth[item["id"]]:diff.append({"arm":arm,"id":item["id"],"expected":truth[item["id"]],
                        "actual":good,"prediction":pred,"gold":rows[item["id"]]["answer"]})
    report={"samples":total,"differences":diff,"extractor":"official MMLU-Pro reading after final </think>"}
    (ROOT/"replication/artifacts/mcq-reference-grading.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=="__main__":main()
