"""Make an exact text-only weight/tokenizer mount for the isolated editor."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]

def main():
    from safetensors import safe_open
    from safetensors.torch import save_file
    base=ROOT/"local_repro/models/base";out=base.parent/"text-editor"
    out.mkdir(exist_ok=True)
    if not (out/"model.safetensors").exists():
        with safe_open(str(base/"model.safetensors"),framework="pt",device="cpu") as f:
            state={n:f.get_tensor(n) for n in f.keys() if n.startswith("model.language_model.")}
        tmp=out/"model.safetensors.tmp";save_file(state,str(tmp));tmp.replace(out/"model.safetensors")
    for name in ("config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja",
                 "special_tokens_map.json","added_tokens.json","vocab.json","merges.txt"):
        if (base/name).exists():shutil.copyfile(base/name,out/name)
    with safe_open(str(base/"model.safetensors"),framework="pt",device="cpu") as a, \
         safe_open(str(out/"model.safetensors"),framework="pt",device="cpu") as b:
        import torch
        assert all(n.startswith("model.language_model.") for n in b.keys())
        for n in b.keys():
            x,y=a.get_tensor(n),b.get_tensor(n)
            assert x.dtype==y.dtype and torch.equal(x,y),n
        report={"text_tensors":len(b.keys()),"values_and_dtypes_exact":True,
                "sha256":hashlib.sha256((out/"model.safetensors").read_bytes()).hexdigest()}
    (out/"export-verification.json").write_text(json.dumps(report,indent=2))
    practice=ROOT/"exp8_math/practice"
    stats={}
    # JSONL is delimited by LF, not Unicode separators such as U+2028.
    # str.splitlines() incorrectly splits one otherwise-valid released example.
    for name,count in {"gsm8k_train":7469,"math_train":7275,"aime":714}.items():
        src=practice/(name+".jsonl")
        with src.open() as f:rows=[json.loads(line) for line in f]
        assert len(rows)==count and len({r["id"] for r in rows})==count
        stats[name]={"examples":len(rows),"sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
            "upstream_bytes_unchanged":True}
    (ROOT/"replication/artifacts/practice-verification.json").write_text(json.dumps(stats,indent=2))
    print(json.dumps(report))

if __name__=="__main__":main()
