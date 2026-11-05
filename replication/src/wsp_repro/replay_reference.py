"""Calibrate evaluator on released policies; never count these as new LLM trials."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .edits import KEYS,logits
from .games import evaluate

ROOT=Path(__file__).resolve().parents[3]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--game",choices=["breakout","seaquest","asterix"],required=True)
    ap.add_argument("--episodes",type=int,default=1000);args=ap.parse_args()
    folder=ROOT/"replication/artifacts/reference-replay";folder.mkdir(parents=True,exist_ok=True)
    file=folder/(args.game+".json")
    if file.exists():raise FileExistsError(file)
    z=np.load(ROOT/"exp5_write_from_scratch/networks"/args.game/"hand.npz")
    tensors={k:z[v] for k,v in KEYS.items()}
    t=time.monotonic();result=evaluate(args.game,lambda o:logits(tensors,o),range(args.episodes))
    reference=json.loads((ROOT/"exp5_write_from_scratch/eval/minatar_eval.json").read_text())[args.game]["hand"]
    result.update(kind="released_network_replay_not_new_generation",published_reference=reference,
                  seconds=time.monotonic()-t,numpy_version=np.__version__)
    file.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}),flush=True)


if __name__=="__main__":main()
