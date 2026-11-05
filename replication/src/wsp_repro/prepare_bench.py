"""Prepare private held-out math sets; never mounted in an editor namespace."""
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[3]
BENCH=ROOT/"replication/artifacts/bench"
SETS={"math500":("HuggingFaceH4/MATH-500","default",500),
      "gsm8k":("openai/gsm8k","main",1319),"amc23":("math-ai/amc23","default",40),
      "minerva":("math-ai/minervamath","default",272),"olympiad":("math-ai/olympiadbench","default",674)}

def fetch(dataset,config,offset):
    query=urllib.parse.urlencode(dict(dataset=dataset,config=config,split="test",offset=offset,length=100))
    for attempt in range(5):
        try:
            with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?"+query,timeout=90) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            if e.code==429 or attempt==4:raise
            time.sleep(2**attempt)
        except Exception:
            if attempt==4:raise
            time.sleep(2**attempt)

def main():
    BENCH.mkdir(parents=True,exist_ok=True);manifest={}
    for name,(dataset,config,count) in SETS.items():
        target=BENCH/(name+".jsonl")
        if not target.exists():
            raw=[]
            for offset in range(0,count,100):
                page=fetch(dataset,config,offset);raw.extend(r["row"] for r in page["rows"])
            assert len(raw)==count,(name,len(raw),count)
            rows=[]
            for i,r in enumerate(raw):
                if name=="math500":row={"id":r["unique_id"],"problem":r["problem"],"answer":r["answer"]}
                elif name=="gsm8k":row={"id":f"gsm8k_test_{i}","problem":r["question"],
                    "answer":r["answer"].split("####")[-1].strip().replace(",","")}
                elif name in ("amc23","minerva"):
                    row={"id":f"amc23_{r['id']}" if name=="amc23" else f"minerva_{i}",
                        "problem":r["question"],"answer":str(r["answer"])}
                else:
                    fa=r["final_answer"]
                    if not isinstance(fa,list):fa=json.loads(fa.replace("'",'"')) if fa.startswith("[") else [fa]
                    row={"id":f"olympiad_{r['id']}","problem":r["question"],"answer":", ".join(map(str,fa))}
                rows.append(row)
            tmp=target.with_suffix(".tmp");tmp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows));tmp.replace(target)
        with target.open() as f:rows=[json.loads(l) for l in f]
        assert len(rows)==count and len({r["id"] for r in rows})==count
        released=set()
        for p in (ROOT/"exp8_math/eval/unedited/parts").glob(name+"*/"+name+"_per_problem.json"):
            released.update(r["id"] for r in json.loads(p.read_text()))
        assert {r["id"] for r in rows}==released,(name,"Released evaluation IDs differ")
        manifest[name]={"dataset":dataset,"config":config,"split":"test","count":count,
            "sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"released_ids_exact":True}
        m=BENCH/"manifest.json";tmp=m.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(m);print(name,count,flush=True)

if __name__=="__main__":
    while True:
        try:main();break
        except urllib.error.HTTPError as e:
            if e.code!=429:raise
            print(json.dumps({"state":"HF_rate_limit_backoff","retry_at":time.time()+1800}),flush=True)
            time.sleep(1800)
