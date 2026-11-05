"""Download GPQA only through authorised HF access; no mirrors or gate bypasses."""
import csv
import hashlib
import json
import random
from pathlib import Path
from .prepare_bench import ROOT,BENCH

def main():
    from huggingface_hub import hf_hub_download
    cache=ROOT/"replication/artifacts/gpqa-source";cache.mkdir(exist_ok=True)
    status=cache/"status.json"
    try:
        src=cache/"gpqa_diamond.csv"
        if not src.exists():src=Path(hf_hub_download("Idavidrein/gpqa","gpqa_diamond.csv",repo_type="dataset",local_dir=cache))
    except Exception as e:
        response=getattr(e,"response",None)
        record={"state":"download_unavailable","error_type":type(e).__name__,
            "http_status":getattr(response,"status_code",None),"automatic_auth":"existing authorised HF cache only",
            "needed":"Authorised GPQA Diamond access or an already legally downloaded official CSV; no credential values in chat"}
        status.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True);return
    with src.open() as f:data=list(csv.DictReader(f))
    expected=set()
    for p in (ROOT/"exp8_math/eval/unedited/parts").glob("gpqa*/gpqa_per_problem.json"):
        expected.update(r["id"] for r in json.loads(p.read_text()))
    rows=[]
    for row in data:
        uid="gpqa_"+row["Record ID"]
        choices=[(True,row["Correct Answer"])]+[(False,row[f"Incorrect Answer {i}"]) for i in range(1,4)]
        seed=int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8],"big")
        random.Random(seed).shuffle(choices)
        answer=chr(65+next(i for i,(good,_) in enumerate(choices) if good))
        prompt="Question: "+row["Question"]+"\nOptions:\n"+"".join(f"{chr(65+i)}. {t}\n" for i,(_,t) in enumerate(choices))
        prompt+='Think step by step and output "The answer is (X)" at the end.'
        rows.append({"id":uid,"kind":"mc","prompt":prompt,"answer":answer})
    assert len(rows)==198 and {r["id"] for r in rows}==expected
    target=BENCH/"gpqa.jsonl";tmp=target.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows));tmp.replace(target)
    p=BENCH/"manifest.json";manifest=json.loads(p.read_text());manifest["gpqa"]={"count":198,"dataset":"Idavidrein/gpqa",
        "split":"gpqa_diamond.csv","source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
        "sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"released_ids_exact":True,
        "option_shuffle":"independent per-ID SHA256 seed; original shuffle helper not released",
        "original_private_prompt":"not released; exact prompt/option order cannot be verified"}
    tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(p)
    status.write_text(json.dumps({"state":"ready","problems":198,"authorised_hf_download":True},indent=2))
    print(json.dumps({"state":"ready","problems":198}),flush=True)

if __name__=="__main__":main()
