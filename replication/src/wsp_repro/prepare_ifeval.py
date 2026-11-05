"""Fetch official IFEval data/checkers at a fixed Google Research commit."""
import hashlib
import json
from pathlib import Path
import urllib.request
from .prepare_bench import ROOT,BENCH

REV="e6890f85757dd84e27ca6df2dd30651dafad28e0"

def main():
    vendor=ROOT/"replication/vendor/ifeval";package=vendor/"instruction_following_eval"
    package.mkdir(parents=True,exist_ok=True);(package/"__init__.py").touch()
    hashes={}
    for name in ["instructions.py","instructions_util.py","instructions_registry.py","data/input_data.jsonl"]:
        target=package/name;target.parent.mkdir(exist_ok=True)
        if not target.exists():
            url=f"https://raw.githubusercontent.com/google-research/google-research/{REV}/instruction_following_eval/{name}"
            with urllib.request.urlopen(url,timeout=90) as f:target.write_bytes(f.read())
        hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    license=vendor/"LICENSE"
    if not license.exists():
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/google-research/google-research/{REV}/LICENSE",timeout=90) as f:
            license.write_bytes(f.read())
    (vendor/"provenance.json").write_text(json.dumps({"repository":"google-research/google-research","commit":REV,"files":hashes},indent=2))
    with (package/"data/input_data.jsonl").open() as f:
        rows=[dict(r,id=f"ifeval_{r['key']}",kind="ifeval") for r in map(json.loads,f)]
    expected=set()
    for p in (ROOT/"exp8_math/eval/unedited/parts").glob("ifeval*/ifeval_per_problem.json"):
        expected.update(r["id"] for r in json.loads(p.read_text()))
    assert len(rows)==541 and {r["id"] for r in rows}==expected
    BENCH.mkdir(exist_ok=True);target=BENCH/"ifeval.jsonl"
    tmp=target.with_suffix(".tmp");tmp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows));tmp.replace(target)
    manifest_path=BENCH/"manifest.json";manifest=json.loads(manifest_path.read_text())
    manifest["ifeval"]={"count":541,"source":"official Google Research IFEval","revision":REV,
        "sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"released_ids_exact":True,
        "checker":"official strict prompt-level; answer after final </think>, empty if none"}
    tmp=manifest_path.with_suffix(".tmp");tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(manifest_path)
    import nltk
    for resource in ["punkt","punkt_tab"]:
        assert nltk.download(resource,download_dir=str(vendor/"nltk_data"),quiet=True)
    print(json.dumps({"ifeval":len(rows),"commit":REV,"original_ids_exact":True}))

if __name__=="__main__":main()
