"""Calibrate released Go weights, then score new frozen networks on available opponents."""
import hashlib
import json
import re
import time
from .go_engine import games,ROOT
from .go_weights import load_policy,SHAPES
from .spec import make_spec
import numpy as np

def main():
    ref=ROOT/"exp6_go_from_scratch/networks";folder=ROOT/"replication/artifacts/go-calibration";folder.mkdir(exist_ok=True)
    path=folder/"released-b0-vs-ppo-s0";score=path/"summary.json"
    if not score.exists():
        games(load_policy(ref/"budget_0"),load_policy(ref/"ppo_same_size_lr3e-5_s0"),2000,31337,path,save_logs=False)
    calibration=json.loads(score.read_text())
    # Independent seeded evaluator streams need not reproduce a published sample exactly.
    # This gate catches observation/color/network/rule mistakes before scoring new policies.
    assert abs(calibration["score"]-.4255)<.07,"Go evaluator calibration is outside tolerance"
    (folder/"validation.json").write_text(json.dumps({"published":.4255,"observed":calibration["score"],
        "episodes":2000,"passed":True,"tolerance":.07,"seed":31337,
        "stream":"our explicit per-game fold-in RNG, original runner omitted"},indent=2))
    jobs=[j for j in make_spec()["jobs"] if j["experiment"]==6]
    while True:
        complete=0
        for job in jobs:
            out=ROOT/"replication/runs"/job["id"];frozen=out/"frozen.json"
            if (out/"evaluation.json").exists():complete+=1;continue
            if not frozen.exists():continue
            final=(out/"final_answer.md").read_text()
            if re.search(r"<(?:python|games|positions)>|```",final):
                (out/"evaluation-held-format.json").write_text(json.dumps({"reason":"Final reply contains an unexecuted code/tool block; inspect controller format handling","resampled":False}))
                continue
            manifest=json.loads(frozen.read_text())
            for name,digest in manifest["tensors"].items():
                assert hashlib.sha256((out/"submission"/(name+".npy")).read_bytes()).hexdigest()==digest
            weights=load_policy(out/"submission");scores={}
            opponents={"random":{k:np.zeros(s,np.float32) for k,s in SHAPES.items()},
                **{f"ppo_same_size_s{i}":load_policy(ref/f"ppo_same_size_lr3e-5_s{i}") for i in range(3)}}
            for opponent,params in opponents.items():
                d=out/"heldout"/opponent
                if not (d/"summary.json").exists():games(weights,params,2000,31337,d,save_logs=False)
                scores[opponent]=json.loads((d/"summary.json").read_text())
            (out/"evaluation.json").write_text(json.dumps({"job":job,"scores":scores,
                "original_hidden_opponents":"unavailable: not released","original_weak_strong_practice":"unavailable: not substituted",
                "scope":"partial reproduction on released same-size PPO policies and random",
                "mean_available_ppo_winrate":np.mean([scores[f"ppo_same_size_s{i}"]["score"] for i in range(3)]).item(),
                "calibration_passed":True,"frozen_tensors":manifest["tensors"]},indent=2))
            complete+=1
        if complete==len(jobs):return
        time.sleep(30)

if __name__=="__main__":main()
