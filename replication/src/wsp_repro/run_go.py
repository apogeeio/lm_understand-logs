"""Independent Go policy writing, with disclosed missing-source opponent limitations."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .providers import call
from .sandbox import run as sandbox
from .go_weights import load_policy,SHAPES
from .spec import make_spec
from .terminal_failure import retained_terminal,terminal_native_output,record_terminal

ROOT=Path(__file__).resolve().parents[3]
NOTICE=("REPRODUCTION INFRASTRUCTURE NOTICE: The authors did not release the weak/strong practice "
    "checkpoints or the smaller hidden evaluation opponents. Random games and positions passes are "
    "available; weak/strong requests fail without spending a pass. No substitute practice opponent is "
    "used. Evaluation will include the three released same-size PPO policies and random; unavailable "
    "original endpoints will be reported unavailable. You cannot access evaluation opponent weights.")

def execute(code,work):
    tree=ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,ast.Attribute) and node.attr in {"backward","autograd","optim","requires_grad_","set_grad_enabled"}:
            return {"returncode":1,"out":"Training/autograd operations are forbidden"}
    return sandbox(code,work,ROOT/".venv-vllm/bin/python",timeout=900,
        prelude="exec(open('/driver.py').read())",readonly=[(Path(__file__).with_name("go_editor.py"),"/driver.py")])

def pass_tool(request,work,pass_id,turn):
    path=turn/"pass.json"
    if path.exists():return json.loads(path.read_text())
    began=time.time()
    result=subprocess.run([str(ROOT/".venv-go/bin/python"),"-m","wsp_repro.go_pass",str(work),str(pass_id)],
        input=json.dumps(request),text=True,capture_output=True,timeout=21600,
        env=dict(os.environ,JAX_PLATFORMS="cpu",OMP_NUM_THREADS="4",PYTHONPATH=str(ROOT/"replication/src")))
    (turn/"pass-stderr.log").write_text(result.stderr[-12000:])
    try:out=json.loads(result.stdout)
    except json.JSONDecodeError:raise RuntimeError("Go pass infrastructure failed; inspect saved stderr")
    out["seconds"]=time.time()-began;path.write_text(json.dumps(out,indent=2));return out

def run(job):
    out=ROOT/"replication/runs"/job["id"];out.mkdir(parents=True,exist_ok=True)
    if (out/"frozen.json").exists() or retained_terminal(out):return
    budget=int(job["case"].split("_")[-1]);work=out/"sandbox";work.mkdir(exist_ok=True)
    task=(ROOT/"exp6_go_from_scratch/runs"/job["case"]/"message.txt").read_text()
    prompt=task+"\n\n"+NOTICE
    (out/"job.json").write_text(json.dumps({**job,"passes":budget,
        "scope":"partial replication: original practice and smaller hidden opponents unreleased",
        "protocol_notice":NOTICE,"original_prompt_sha256":hashlib.sha256(task.encode()).hexdigest()},indent=2))
    history=[{"role":"user","content":prompt}];used=0;python_runs=0
    # Replaying saved replies and tool outputs reconstructs history without rerunning code or passes.
    for turn in range(420):
        td=out/f"turn-{turn:03d}";td.mkdir(exist_ok=True)
        if (td/"call.json").exists() and (td/"answer.md").exists():
            meta=json.loads((td/"call.json").read_text());text=(td/"answer.md").read_text()
        else:text,meta=call(job["model"],job["effort"],history,td)
        if meta["transport"] in ("openai_api","coding_plan_responses_bridge"):history.extend(meta["continuation_items"])
        else:history.append({"role":"assistant","content":text})
        blocks=re.findall(r"```(python|games|positions)\s*\n(.*?)```",text,re.S)
        if len(blocks)>1:feedback="Reply with exactly one python, games or positions block. Nothing was run."
        elif blocks:
            kind,code=blocks[0]
            if kind=="python":
                if python_runs>=120:feedback="The 120 Python-run budget is exhausted. Freeze saved tensors and finish."
                else:
                    cache=td/"python.json"
                    try:result=json.loads(cache.read_text()) if cache.exists() else execute(code,work)
                    except (SyntaxError,ValueError) as e:result={"returncode":1,"out":str(e)}
                    cache.write_text(json.dumps(result,indent=2));python_runs+=1;feedback=result["out"]
            elif used>=budget:feedback="No passes remain. Finalize saved tensors and reply without a block."
            else:
                try:
                    req=json.loads(code);req["type"]=kind;result=pass_tool(req,work,used+1,td)
                    if result["ok"]:used+=1;feedback=json.dumps(result["result"])
                    else:feedback="Pass failed, no budget spent: "+result["message"]
                except (json.JSONDecodeError,ValueError,TypeError) as e:feedback="Invalid pass request; no pass spent: "+str(e)
        elif re.search(r"<(?:python|games|positions)>|```",text):
            feedback=("No tool ran. XML tags and other code-fence labels are not supported. "
                "Use exactly one Markdown fenced block opening with three backticks immediately followed "
                +f"by python, games or positions. {used}/{budget} passes used; this reply spent no pass.")
        elif used<budget:feedback=f"{used}/{budget} passes used. Continue until all allocated passes are used, then finalize. "+NOTICE
        else:
            tensors=load_policy(work/"submission");submission=out/"submission";submission.mkdir(exist_ok=True);hashes={}
            for name,value in tensors.items():
                p=submission/(name+".npy");np_save(p,value);hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            frozen={"job":job,"transport":meta["transport"],"tensors":hashes,"passes_used":used,
                "python_runs":python_runs,"frozen_before_evaluation":True,"semantic_protocol_audit":"pending_trace_review",
                "scope":"partial replication: unavailable original practice/hidden opponents",
                "original_practice_checkpoint_substitution":False}
            (out/"final_answer.md").write_text(text);(out/"frozen.json").write_text(json.dumps(frozen,indent=2));return
        history.append({"role":"user","content":feedback[:20000]})
        (out/"search-state.json").write_text(json.dumps({"turn":turn+1,"passes_used":used,"python_runs":python_runs,
            "transport":meta["transport"],"updated":time.time()}))
    raise RuntimeError("Go turn limit reached; incomplete attempt retained")

def np_save(path,value):
    import numpy as np
    np.save(path,value)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=["astra","deepseek"],required=True);args=ap.parse_args()
    for job in make_spec()["jobs"]:
        if job["experiment"]!=6 or job["model"]!=args.model:continue
        while True:
            try:run(job);break
            except Exception as e:
                p=ROOT/"replication/runs"/job["id"];p.mkdir(parents=True,exist_ok=True)
                error={"type":type(e).__name__,"message":str(e),"updated":time.time()}
                (p/"search-error.json").write_text(json.dumps(error));print(json.dumps(error),flush=True)
                if terminal_native_output(error):
                    record_terminal(p,error);break
                if type(e).__name__ in {"RateLimited","TransientProviderError","QuotaExhausted","URLError","TimeoutError"}:
                    time.sleep(1800);continue
                raise

if __name__=="__main__":main()
