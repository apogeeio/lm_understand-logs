"""Detach study services without duplicating a live process; run on the GPU host."""
from .execution_scope import math_paused
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import fcntl

ROOT=Path(__file__).resolve().parents[3]

def launch(name,module,extra=(),gpu=False,python_env=".venv-vllm",env_extra=None):
    if module in {"wsp_repro.run_math", "wsp_repro.math_gpu", "wsp_repro.math_evaluate"} and math_paused(ROOT):
        print(json.dumps({"service":name,"state":"paused"}));return
    folder=ROOT/"replication/artifacts/control";folder.mkdir(parents=True,exist_ok=True)
    with (folder/(name+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Also detects manually launched instances, not only those created by this launcher.
        for p in Path("/proc").glob("[0-9]*/cmdline"):
            try:cmd=p.read_bytes().decode().rstrip("\0").split("\0")
            except (OSError,UnicodeError):continue
            if "-m" not in cmd:continue
            i=cmd.index("-m")
            if cmd[i+1:]==[module,*extra] and "python" in cmd[0] and p.parent.name!=str(os.getpid()):
                print(json.dumps({"service":name,"state":"already_running","pid":int(p.parent.name)}));return
        env=os.environ.copy();env["PYTHONPATH"]=str(ROOT/"replication/src");env["OMP_NUM_THREADS"]="4"
        env["PATH"]=str(ROOT/python_env/"bin")+":"+env["PATH"]
        if env_extra:env.update(env_extra)
        if gpu:
            cuda=ROOT/".venv-vllm/lib/python3.12/site-packages/nvidia/cu13"
            if not (cuda/"lib64").exists():(cuda/"lib64").symlink_to("lib",target_is_directory=True)
            if not (cuda/"lib/libcudart.so").exists():(cuda/"lib/libcudart.so").symlink_to("libcudart.so.13")
            env["CUDA_HOME"]=str(cuda);env["PATH"]=str(cuda/"bin")+":"+env["PATH"]
        log=folder/(name+"-"+time.strftime("%Y%m%dT%H%M%S")+".log")
        with log.open("a") as f:
            p=subprocess.Popen([str(ROOT/python_env/"bin/python"),"-u","-m",module,*extra],cwd=ROOT,
                env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record={"service":name,"pid":p.pid,"log":str(log),"started":time.time()}
        (folder/(name+"-launch.json")).write_text(json.dumps(record,indent=2));print(json.dumps(record))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("service",choices=["gpu","astra","deepseek"])
    args=ap.parse_args()
    if args.service=="gpu":launch("math-gpu","wsp_repro.math_gpu",gpu=True)
    else:launch("math-search-"+args.service,"wsp_repro.run_math",("--model",args.service))

if __name__=="__main__":main()
