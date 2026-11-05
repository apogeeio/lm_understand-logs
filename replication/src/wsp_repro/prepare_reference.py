"""Freeze the published p50 patch as a calibration arm, never a new model generation."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]

def main():
    import torch
    source=ROOT/"exp8_math/runs/p50c/edit";out=ROOT/"replication/runs/eval-released-p50"
    target=out/"sandbox/edited";target.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/"exp8_math/code"))
    import apply_edit
    apply_edit.main(str(ROOT/"local_repro/models/base/model.safetensors"),str(source),str(target))
    manifest=json.loads((source/"manifest.json").read_text());hashes={}
    for row in manifest["tensors"]:
        p=target/(row["name"]+".npy");a=np.load(p,allow_pickle=False)
        bits=torch.from_numpy(a).to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
        assert hashlib.sha256(bits).hexdigest()==row["edited_sha256"]
        hashes[row["name"]]=hashlib.sha256(p.read_bytes()).hexdigest()
    job={"id":"eval-released-p50","model":"released_reference","experiment":8,
        "calibration_only":True,"counts_towards_independent_generations":False}
    (out/"frozen.json").write_text(json.dumps({"job":job,"edited":hashes,"frozen_before_evaluation":True,
        "source":"published exact p50 patch","source_tensor_hashes_verified":True},indent=2))
    print(json.dumps({"published_patch_verified":True,"tensors":len(hashes),"calibration_only":True}),flush=True)

if __name__=="__main__":main()
