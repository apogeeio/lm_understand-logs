"""Independent full edit-search trajectories; no evaluation examples enter the editor sandbox."""
from .execution_scope import math_paused
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import select
import subprocess
import time
import uuid
from .providers import call
from .spec import make_spec
from .terminal_failure import retained_terminal,terminal_native_output,record_terminal

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/"local_repro/models/base"
EDITOR_BASE=ROOT/"local_repro/models/text-editor"
QUEUE=ROOT/"replication/artifacts/math-forward-queue"
NOTICE_TURNS={10:(52,59,65),25:(16,25,31),50:(29,46,51)}

def explanation():
    text=(ROOT/"exp8_math/runs/p10c/message.txt").read_text()
    return re.search(r"EXPLAIN EVERY STEP\..*?(?=\n\nRULES)",text,re.S).group()


class Kernel:
    def __init__(self,work):
        self.work=work;self.proc=None;self.start()

    def start(self):
        python=ROOT/".venv-vllm/bin/python"
        info=json.loads(subprocess.check_output([str(python),"-c",
            "import sys,sysconfig,json;print(json.dumps([sys.prefix,sys.base_prefix,sysconfig.get_paths()['purelib']]))"],text=True))
        cmd=["bwrap","--unshare-all","--die-with-parent","--new-session","--clearenv",
            "--ro-bind","/usr","/usr","--symlink","usr/lib","/lib","--symlink","usr/lib64","/lib64",
            "--symlink","usr/bin","/bin","--proc","/proc","--dev","/dev","--tmpfs","/tmp"]
        for path in dict.fromkeys(info[:2]):cmd += ["--ro-bind",path,path]
        if not (EDITOR_BASE/"model.safetensors").exists():
            raise RuntimeError("Prepare the text-only editor export before starting a search")
        cmd += ["--ro-bind",str(EDITOR_BASE),"/model","--ro-bind",str(ROOT/"exp8_math/practice"),"/practice",
            "--ro-bind",str(Path(__file__).with_name("math_kernel.py")),"/driver.py",
            "--bind",str(self.work),"/work","--chdir","/work","--setenv","PYTHONPATH",info[2],
            "--setenv","CUDA_VISIBLE_DEVICES","","--setenv","OMP_NUM_THREADS","4",
            "--setenv","TRITON_CACHE_DIR","/tmp/triton","--setenv","MPLCONFIGDIR","/tmp/mpl",
            str(python.resolve()),"-u","/driver.py"]
        def limits():
            import resource
            resource.setrlimit(resource.RLIMIT_AS,(96<<30,96<<30))
        self.stderr=open(self.work.parent/"kernel-stderr.log","a")
        self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
            text=True,bufsize=1,preexec_fn=limits,env={"PATH":"/usr/bin:/bin"})

    def run(self,code):
        self.proc.stdin.write(json.dumps({"code":code})+"\n");self.proc.stdin.flush()
        ready,_,_=select.select([self.proc.stdout],[],[],920)
        if not ready:
            self.close();self.start()
            return {"ok":False,"out":"Kernel exceeded 900 seconds and was restarted; variables were lost; files persist."}
        line=self.proc.stdout.readline()
        if not line:
            self.close();self.start()
            return {"ok":False,"out":"Kernel exited and was restarted; variables were lost; files persist."}
        return json.loads(line)

    def close(self):
        if self.proc and self.proc.poll() is None:self.proc.terminate();self.proc.wait(timeout=10)
        if hasattr(self,"stderr"):self.stderr.close()


def forward(request,work,pass_id,turn_dir):
    if request.get("type") not in ("generate","teacher_forced"):
        raise ValueError("Editor may only request practice generation or teacher-forced passes")
    previous=turn_dir/"forward.json"
    if previous.exists():return json.loads(previous.read_text())
    QUEUE.mkdir(parents=True,exist_ok=True)
    receipt=turn_dir/"forward-request.json"
    if receipt.exists():uid=json.loads(receipt.read_text())["id"]
    else:
        uid=work.parent.name+"-"+uuid.uuid4().hex
        receipt.write_text(json.dumps({"id":uid}))
    path=QUEUE/(uid+".request.json");tmp=path.with_suffix(".tmp")
    envelope={"work":str(work),"pass_id":pass_id,"request":request}
    response=QUEUE/(uid+".response.json")
    processing=path.with_suffix(".processing")
    if not any(p.exists() for p in (path,processing,response)):
        tmp.write_text(json.dumps(envelope));tmp.replace(path)
    started=time.monotonic()
    while not response.exists():
        if time.monotonic()-started>21600:raise TimeoutError("Forward worker exceeded six hours; request retained")
        time.sleep(2)
    result=json.loads(response.read_text());(turn_dir/"forward.json").write_text(json.dumps(result,indent=2))
    return result


def run(job):
    if math_paused(ROOT):return
    scope=ROOT/"replication/configs/search-deferrals.json"
    if scope.exists() and job["passes"] in json.loads(scope.read_text())["math_pass_budgets"]:return
    out=ROOT/"replication/runs"/job["id"];out.mkdir(parents=True,exist_ok=True)
    if (out/"frozen.json").exists() or retained_terminal(out):return
    state_path=out/"search-state.json"
    if state_path.exists():
        state=json.loads(state_path.read_text());history=state["history"];turn=state["turn"]
        used=state["passes_used"];python_runs=state["python_runs"]
        unexplained=state.get("unexplained_streak",0)
        history.append({"role":"user","content":"Infrastructure resume: Python kernel restarted, variables were lost. Saved files and edits persist. Reload any variables you need."})
    else:
        run_name={10:"p10c",25:"p25c",50:"p50c"}[job["passes"]]
        task=(ROOT/"exp8_math/runs"/run_name/"turns/t001/prompt.txt").read_text()
        history=[{"role":"user","content":task}];turn=0;used=0;python_runs=0;unexplained=0
    task=(ROOT/"exp8_math/runs"/{10:"p10c",25:"p25c",50:"p50c"}[job["passes"]]/"turns/t001/prompt.txt").read_text()
    work=out/"sandbox";work.mkdir(exist_ok=True)
    if not (work/"eval_prompt_format.txt").exists():
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(EDITOR_BASE,local_files_only=True)
        prompt="<problem>\nPlease reason step by step, and put your final answer within \\boxed{}."
        rendered=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,
            add_generation_prompt=True,enable_thinking=True)
        (work/"eval_prompt_format.txt").write_text("Exact template (replace <problem> with the question):\n"+rendered)
    (out/"job.json").write_text(json.dumps(job,indent=2))
    kernel=Kernel(work)
    try:
        while turn<420:
            ex,sz,facts=NOTICE_TURNS[job["passes"]]
            notice={ex:explanation(),sz:(ROOT/"exp8_math/notices/001_pass_size.txt").read_text(),
                    facts:(ROOT/"exp8_math/notices/002_tool_facts.txt").read_text()}.get(turn+1)
            if notice:history.append({"role":"user","content":notice})
            td=out/f"turn-{turn:03d}";td.mkdir(exist_ok=True)
            if (td/"call.json").exists() and (td/"answer.md").exists():
                metadata=json.loads((td/"call.json").read_text());text=(td/"answer.md").read_text()
            else:text,metadata=call(job["model"],job["effort"],history,td)
            if metadata["transport"] in ("openai_api","coding_plan_responses_bridge"):
                history.extend(metadata["continuation_items"])
            else:history.append({"role":"assistant","content":text})
            blocks=re.findall(r"```(python|forward)\s*\n(.*?)```",text,re.S)
            needs_explanation=turn+1>=ex and blocks and len(text.split("```",1)[0].strip())<40
            if needs_explanation and unexplained<2:
                unexplained+=1;feedback=explanation()+"\nYour block was not run. Explain this step before sending it again."
            elif blocks:
                unexplained=0
                if len(blocks)!=1:feedback="Please reply with exactly one python or forward block."
                else:
                    kind,code=blocks[0]
                    if kind=="python":
                        if python_runs>=300:feedback="The 300 Python-run budget is exhausted. Select the saved edit and finish."
                        else:
                            saved=td/"python.json"
                            result=json.loads(saved.read_text()) if saved.exists() else kernel.run(code)
                            python_runs+=1
                            saved.write_text(json.dumps({"code":code,**result},indent=2))
                            feedback=result["out"] or "(no output)"
                    elif used>=job["passes"]:feedback="The forward-pass budget is exhausted. Freeze your best edit and finish."
                    else:
                        try:
                            request=json.loads(code);result=forward(request,work,used+1,td)
                            if result["ok"]:used+=1;feedback=json.dumps(result["result"])
                            else:feedback="Forward request failed; does not count: "+result.get("message","")
                        except (ValueError,TypeError) as e:feedback="Invalid forward request, no pass spent: "+str(e)
            elif "<forward>" in text or "<python>" in text or "```" in text:
                feedback=("No tool ran. XML tags and other code-fence labels are not recognized. "
                    "Use exactly one Markdown fenced block, opening with three backticks immediately "
                    "followed by forward (JSON request) or python (code), as specified in the task. "
                    +f"{used}/{job['passes']} forward passes used; no pass was spent on this reply.")
            elif used<job["passes"]:
                feedback=f"{used} of {job['passes']} forward passes used. Continue until the allocated passes are used, then finalize."
            else:
                from .math_weights import validate
                names=validate(BASE,work/"edited")
                frozen={"job":job,"edited":{},"passes_used":used,"python_runs":python_runs,
                    "transport":metadata["transport"],"semantic_protocol_audit":"pending_trace_review",
                    "heldout_evaluation":"pending","frozen_before_evaluation":True}
                for name in names:
                    p=work/"edited"/(name+".npy");frozen["edited"][name]=hashlib.sha256(p.read_bytes()).hexdigest()
                (out/"final_answer.md").write_text(text)
                (out/"frozen.json").write_text(json.dumps(frozen,indent=2));return
            history.append({"role":"user","content":feedback[:10000].encode("ascii","replace").decode()})
            turn+=1
            if turn%40==0:
                notes=(work/"notes.md").read_text()[-30000:] if (work/"notes.md").exists() else "No notes saved."
                saved=sorted(p.name for p in (work/"edited").glob("*.npy"))
                history=[{"role":"user","content":task+"\n\nContinuation of this same isolated run. "
                    +f"{used}/{job['passes']} passes and {python_runs}/300 Python runs used. "
                    +"Kernel/files persist. Current edits: "+json.dumps(saved)+"\nYour saved notes:\n"+notes
                    +"\nLatest controlled output:\n"+feedback[:10000]}]
                if turn>=ex:history.append({"role":"user","content":explanation()})
            state_path.write_text(json.dumps({"turn":turn,"passes_used":used,"python_runs":python_runs,
                "unexplained_streak":unexplained,"history":history}))
        raise RuntimeError("Turn budget exhausted; incomplete run retained")
    finally:kernel.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=["astra","deepseek"],required=True)
    args=ap.parse_args()
    # Expensive searches are independent and sequential within each model; the GPU forward queue is shared.
    for job in make_spec()["jobs"]:
        if job["experiment"]==8 and job["model"]==args.model:
            while True:
                try:run(job);break
                except Exception as e:
                    folder=ROOT/"replication/runs"/job["id"]
                    folder.mkdir(parents=True,exist_ok=True)
                    error={"type":type(e).__name__,"message":str(e),"updated":time.time()}
                    (folder/"search-error.json").write_text(json.dumps(error))
                    print(json.dumps({"job":job["id"],"error":error}),flush=True)
                    if terminal_native_output(error):
                        record_terminal(folder,error);break
                    if type(e).__name__ in {"RateLimited","TransientProviderError","QuotaExhausted","URLError","TimeoutError"}:
                        time.sleep(1800);continue
                    raise


if __name__=="__main__":main()
