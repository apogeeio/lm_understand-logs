"""Trusted CPU process for Go feedback; original missing opponents are never substituted."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .go_engine import games,positions,ROOT
from .go_weights import load_policy,SHAPES

def main():
    ap=argparse.ArgumentParser();ap.add_argument("work");ap.add_argument("pass_id",type=int)
    args=ap.parse_args();work=Path(args.work).resolve()
    if (ROOT/"replication/runs").resolve() not in work.parents:raise ValueError("Invalid work root")
    request=json.load(sys.stdin);kind=request["type"];weights=load_policy(work/"submission")
    folder=work/f"pass_{args.pass_id}";folder.mkdir(exist_ok=True)
    if kind=="positions":result={"top3":positions(weights,request,folder),"folder":"/work/"+folder.name}
    elif kind=="games":
        counts=request.get("games",{})
        if not counts or set(counts)-{"random","weak","strong"}:raise ValueError("Unknown practice opponent")
        if any(type(n) is not int or n<0 or n%2 for n in counts.values()) or not 2<=sum(counts.values())<=128:
            raise ValueError("Games pass requires even counts, total2..128")
        missing=[o for o in ("weak","strong") if counts.get(o,0)]
        if missing:raise ValueError("Original practice checkpoints not released: "+", ".join(missing)+". Request random games or positions; no pass spent.")
        random={n:np.zeros(s,np.float32) for n,s in SHAPES.items()}
        result={"random":games(weights,random,counts["random"],17000+args.pass_id,folder),
            "folder":"/work/"+folder.name}
        path=folder/"games.json";records=json.loads(path.read_text())
        for row in records:row["opponent"]="random"
        path.write_text(json.dumps(records));(folder/"summary.json").write_text(json.dumps({"random":result["random"]},indent=2))
        (folder/"README.txt").write_text("games.json contains opponent, net_color, result, margin, end, captures/losses and moves "
            "(ply/by/move/top5 for network turns). boards.npy=(games,162,81), int8 board after each ply. "
            "valid.npy=(games,162), true for played plies. X/Black=1, O/White=-1, empty=0.")
    else:raise ValueError("Unknown Go pass type")
    print(json.dumps({"ok":True,"result":result}),flush=True)

if __name__=="__main__":
    try:main()
    except Exception as e:print(json.dumps({"ok":False,"error_type":type(e).__name__,"message":str(e)}),flush=True)
