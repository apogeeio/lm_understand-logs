"""Dispatch frozen game artifacts to the authorised offline host; no credentials are copied."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from .sandbox import run
from .settings import connection

ROOT=Path(__file__).resolve().parents[3]


def evaluate_frozen(job):
    out=ROOT/"replication/runs"/job["id"]
    if (out/"evaluation.json").exists():return
    metadata=json.loads((out/"artifact.json").read_text())
    host,remote_root=connection()
    remote=remote_root+"/replication/runs/"+job["id"]
    local=Path(remote_root).resolve()==ROOT.resolve()
    if not local:
        subprocess.run(["ssh","-o","BatchMode=yes",host,shlex.join(["mkdir","-p",remote])],check=True)
        subprocess.run(["rsync","-az",str(out/metadata["artifact"]),host+":"+remote+"/"],check=True)
    if job["kind"]=="python":
        cmd=[remote_root+"/.venv-games/bin/python","-m","wsp_repro.dispatch_eval","--sandbox-job",job["id"],
             "--game",job["game"],"--episodes",str(job["episodes"])]
    else:
        cmd=[remote_root+"/.venv-games/bin/python","-m","wsp_repro.evaluate_game","--game",job["game"],
            "--network",remote+"/network.npz","--episodes",str(job["episodes"]),"--output",remote+"/evaluation.json"]
    if local:
        result=subprocess.run(cmd,text=True,capture_output=True,timeout=3700,
            env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONPATH=str(ROOT/"replication/src")))
    else:
        result=subprocess.run(["ssh","-o","BatchMode=yes",host,
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="+shlex.quote(remote_root+"/replication/src")+" "+shlex.join(cmd)],
            text=True,capture_output=True,timeout=3700)
    (out/"eval-launch.json").write_text(json.dumps({"returncode":result.returncode,
        "stdout":result.stdout[-10000:],"stderr":result.stderr[-6000:]},indent=2))
    if result.returncode:raise RuntimeError("Evaluation failed; inspect eval-launch.json")
    if not local:
        subprocess.run(["rsync","-az",host+":"+remote+"/evaluation.json",str(out/"evaluation.json")],check=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--sandbox-job",required=True)
    ap.add_argument("--game",required=True);ap.add_argument("--episodes",type=int,default=1000)
    args=ap.parse_args()
    if not all(c.isalnum() or c in "-_" for c in args.sandbox_job):ap.error("Invalid ID")
    folder=ROOT/"replication/runs"/args.sandbox_job;work=folder/"eval_sandbox";work.mkdir(exist_ok=True)
    shutil.copyfile(folder/"policy.py",work/"policy.py")
    code="""import sys
sys.path.insert(0,'/package')
from wsp_repro.evaluate_game import main
sys.argv=['evaluate_game','--game',GAME,'--code','/work/policy.py','--inside-sandbox',
          '--episodes',EPISODES,'--output','/work/evaluation.json']
main()
""".replace("GAME",repr(args.game)).replace("EPISODES",repr(str(args.episodes)))
    result=run(code,work,ROOT/".venv-games/bin/python",timeout=3600,
               readonly=[(ROOT/"replication/src","/package")])
    (folder/"sandbox-evaluation.json").write_text(json.dumps(result,indent=2))
    if result["returncode"]:raise SystemExit(result["returncode"])
    shutil.copyfile(work/"evaluation.json",folder/"evaluation.json")
    print(result["out"])


if __name__=="__main__":main()
