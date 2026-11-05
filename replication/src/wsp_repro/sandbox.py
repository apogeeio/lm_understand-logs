"""Linux CPU sandbox. Only task inputs/work are bound; no host home, keys, network or GPU."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[3]


def run(code,work,python,prelude="",timeout=600,readonly=(),extra_sites=()):
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    python=Path(python).absolute()
    info=json.loads(subprocess.check_output([str(python),"-c",
        "import json,sys,sysconfig;print(json.dumps([sys.prefix,sys.base_prefix,sysconfig.get_paths()['purelib']]))"],text=True))
    cmd=["bwrap","--unshare-all","--die-with-parent","--new-session","--clearenv",
        "--ro-bind","/usr","/usr","--symlink","usr/lib","/lib","--symlink","usr/lib64","/lib64",
        "--symlink","usr/bin","/bin","--proc","/proc","--dev","/dev","--tmpfs","/tmp"]
    for path in dict.fromkeys([*info[:2],*map(str,extra_sites)]):cmd += ["--ro-bind",path,path]
    for source,target in readonly:cmd += ["--ro-bind",str(Path(source).resolve()),target]
    paths=":".join([info[2],*map(str,extra_sites)])
    cmd += ["--bind",str(work),"/work","--chdir","/work","--setenv","PATH","/usr/bin",
        "--setenv","MPLCONFIGDIR","/tmp/matplotlib","--setenv","CUDA_VISIBLE_DEVICES","",
        "--setenv","PYTHONPATH",paths,"--setenv","OMP_NUM_THREADS","1",
        "--setenv","OPENBLAS_NUM_THREADS","1",str(python.resolve()),"-u","-"]
    def limits():
        resource.setrlimit(resource.RLIMIT_AS,(96<<30,96<<30))
        resource.setrlimit(resource.RLIMIT_CPU,(timeout,timeout+5))
    try:
        p=subprocess.run(cmd,input=prelude+"\n"+code,text=True,capture_output=True,timeout=timeout,
                         preexec_fn=limits,env={"PATH":"/usr/bin:/bin"})
        out=p.stdout+("\n[stderr]\n"+p.stderr[-6000:] if p.stderr else "")
        return {"returncode":p.returncode,"out":out[:20000],"output_truncated":len(out)>20000}
    except subprocess.TimeoutExpired:return {"returncode":124,"out":"Execution timed out"}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--job",required=True);ap.add_argument("--language",required=True)
    args=ap.parse_args()
    if not all(c.isalnum() or c in "-_" for c in args.job):ap.error("Invalid job ID")
    if args.language not in ("python","romaji","english"):ap.error("Invalid corpus")
    work=ROOT/"replication/runs"/args.job/("sandbox-"+uuid.uuid4().hex)
    source=ROOT/"exp3_language/networks"/args.language/"blocks.npz"
    prelude="import numpy as np\nW=dict(np.load('/inputs/blocks.npz',allow_pickle=False))\n"
    python=ROOT/".venv-vllm/bin/python"
    if not python.exists():python=ROOT/".venv-games/bin/python"
    response=run(sys.stdin.read(),work,python,prelude,
        readonly=[(source,"/inputs/blocks.npz")])
    print(json.dumps(response),flush=True)


if __name__=="__main__":main()
