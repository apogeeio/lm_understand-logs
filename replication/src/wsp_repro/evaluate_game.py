"""CPU evaluation entry point. Model-generated Python must use --inside-sandbox."""
import argparse
import json
from pathlib import Path
import signal
import numpy as np
from .edits import KEYS,logits
from .games import evaluate
from .code_policy import extract


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--game",required=True)
    ap.add_argument("--network",type=Path)
    ap.add_argument("--code",type=Path)
    ap.add_argument("--inside-sandbox",action="store_true")
    ap.add_argument("--episodes",type=int,default=1000)
    ap.add_argument("--max-steps",type=int,default=10000)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if bool(args.network)==bool(args.code):ap.error("Choose network or code")
    if args.network:
        z=np.load(args.network,allow_pickle=False)
        tensors={name:z[name] if name in z else z[key] for name,key in KEYS.items()}
        policy=lambda obs:logits(tensors,obs)
    else:
        if not args.inside_sandbox:ap.error("Python policy evaluation requires the external Linux sandbox")
        source,report=extract("```python\n"+args.code.read_text()+"\n```")
        namespace={};exec(compile(source,"<policy>","exec"),namespace)
        act=namespace["act"]
        def timed_out(signum,frame):raise TimeoutError("act(obs) exceeded 0.1s")
        signal.signal(signal.SIGALRM,timed_out)
        def policy(obs):
            signal.setitimer(signal.ITIMER_REAL,0.1)
            try:return act(obs.copy())
            finally:signal.setitimer(signal.ITIMER_REAL,0)
    result=evaluate(args.game,policy,range(args.episodes),args.max_steps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}),flush=True)


if __name__=="__main__":main()
