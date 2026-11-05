"""Calibrate the car physics, then evaluate each single frozen repair without feedback."""
import hashlib
import json
import time
import numpy as np
from .car import ROOT,evaluate,rounded_actor
from .spec import make_spec

def score(actor,n,seed,path):
    if path.exists():return json.loads(path.read_text())
    return evaluate(actor,n,seed,path)

def main():
    folder=ROOT/"replication/artifacts/car-calibration";folder.mkdir(exist_ok=True)
    base=score(rounded_actor(),256,1400,folder/"rounded-base.json")
    ref=score(dict(np.load(ROOT/"exp4_fix_car/networks/edited_actor.npz")),1024,1401,folder/"released-repair.json")
    assert base["success"]==0 and ref["success"]==1,"Car calibration differs from released endpoints"
    jobs=[j for j in make_spec()["jobs"] if j["experiment"]==4]
    while True:
        complete=0
        for job in jobs:
            out=ROOT/"replication/runs"/job["id"]
            if (out/"evaluation.json").exists():complete+=1;continue
            path=out/"artifact.json"
            if not path.exists():continue
            artifact=json.loads(path.read_text());actor=out/artifact["artifact"]
            assert hashlib.sha256(actor.read_bytes()).hexdigest()==artifact["sha256"],"Frozen repair changed"
            weights=dict(np.load(actor))
            search=score(weights,256,1400,out/"search-evaluation.json")
            test=score(weights,1024,1401,out/"test-evaluation.json")
            (out/"evaluation.json").write_text(json.dumps({"job":job,"search":search,"test":test,
                "frozen_sha256":artifact["sha256"],"calibration_passed":True,
                "search_seed":1400,"test_seed":1401,"seed_stream":"our fixed streams; original seeds not released"},indent=2))
            complete+=1
        if complete==len(jobs):return
        time.sleep(30)

if __name__=="__main__":main()
