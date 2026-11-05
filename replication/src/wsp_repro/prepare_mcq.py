"""Recover the released MMLU-Pro subset and use its official few-shot CoT format."""
import hashlib
import json
from pathlib import Path
import urllib.request
from .prepare_bench import ROOT,BENCH

def fetch(url,path):
    if not path.exists():
        with urllib.request.urlopen(url,timeout=120) as f:path.write_bytes(f.read())

def example(row,cot=""):
    cot=cot or "Let's think step by step."
    if cot.startswith("A: "):cot=cot[3:]
    options=[o for o in row["options"] if o!="N/A"]
    return "Question: "+row["question"]+"\nOptions: "+"".join(f"{chr(65+i)}. {o}\n" for i,o in enumerate(options))+"Answer: "+cot+"\n\n"

def main():
    import pyarrow.parquet as pq
    cache=ROOT/"replication/artifacts/mcq-source";cache.mkdir(exist_ok=True)
    meta=cache/"mmlupro-source.json"
    if not meta.exists():
        with urllib.request.urlopen("https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro",timeout=60) as f:info=json.load(f)
        with urllib.request.urlopen("https://api.github.com/repos/TIGER-AI-Lab/MMLU-Pro/commits?path=evaluate_from_api.py&per_page=1",timeout=60) as f:code=json.load(f)
        meta.write_text(json.dumps({"dataset_revision":info["sha"],"official_prompt_commit":code[0]["sha"]},indent=2))
    info=json.loads(meta.read_text());revision=info["dataset_revision"]
    for split in ("test","validation"):
        fetch(f"https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/{revision}/data/{split}-00000-of-00001.parquet",cache/(split+".parquet"))
    fetch(f"https://raw.githubusercontent.com/TIGER-AI-Lab/MMLU-Pro/{info['official_prompt_commit']}/evaluate_from_api.py",cache/"official_evaluate_from_api.py")
    validation=pq.read_table(cache/"validation.parquet").to_pylist();data=pq.read_table(cache/"test.parquet").to_pylist()
    expected=set()
    for p in (ROOT/"exp8_math/eval/unedited/parts").glob("mmlupro1k*/mmlupro1k_per_problem.json"):
        expected.update(r["id"] for r in json.loads(p.read_text()))
    rows=[]
    for row in data:
        uid=f"mmlupro_{row['question_id']}"
        if uid not in expected:continue
        category=row["category"]
        prompt=(f"The following are multiple choice questions (with answers) about {category}. Think step by"
            ' step and then output the answer in the format of "The answer is (X)" at the end.\n\n')
        examples=[r for r in validation if r["category"]==category]
        prompt+="".join(example(r,r["cot_content"]) for r in examples)+example(row)
        rows.append({"id":uid,"kind":"mc","prompt":prompt,"answer":row["answer"],"category":category,
            "validation_examples":len(examples)})
    assert len(rows)==1000 and {r["id"] for r in rows}==expected
    BENCH.mkdir(exist_ok=True);target=BENCH/"mmlupro1k.jsonl"
    tmp=target.with_suffix(".tmp");tmp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows));tmp.replace(target)
    manifest_path=BENCH/"manifest.json";manifest=json.loads(manifest_path.read_text())
    manifest["mmlupro1k"]={"count":1000,"dataset":"TIGER-Lab/MMLU-Pro","revision":revision,
        "sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"released_ids_exact":True,
        "prompt_source":"TIGER-AI-Lab/MMLU-Pro evaluate_from_api.py","prompt_commit":info["official_prompt_commit"],
        "private_original_prompt_helper":"not released; exact match cannot be verified",
        "format":"official per-category validation CoT examples, followed by test question"}
    tmp=manifest_path.with_suffix(".tmp");tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(manifest_path)
    print(json.dumps(manifest["mmlupro1k"]),flush=True)

if __name__=="__main__":main()
