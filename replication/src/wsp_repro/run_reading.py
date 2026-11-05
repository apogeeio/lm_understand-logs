"""Fresh calls for experiments 1/2; controlled Python conversation for experiment 3."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from .spec import make_spec
from .providers import call
from .settings import connection

ROOT=Path(__file__).resolve().parents[3]


def prompt_for(job):
    exp=job["experiment"];case=job["case"]
    folder={1:"exp1_which_game",2:"exp2_diffusion",3:"exp3_language"}[exp]
    run=case+"_r0" if exp==3 else case
    name="message.txt" if exp==3 else "prompt.txt"
    return (ROOT/folder/"runs"/run/name).read_text()


def section(text,label):
    lines=[re.sub(r"[*_`]","",line).strip() for line in text.splitlines()]
    for i,line in enumerate(lines):
        line=re.sub(r"^#{1,6}\s*","",line)
        m=re.fullmatch(re.escape(label)+r"\s*(?::\s*(.*))?",line,re.I)
        if not m:continue
        if m.group(1):return m.group(1).strip().lower()
        content=[]
        for following in lines[i+1:]:
            if not following:
                if content:break
                continue
            if re.match(r"^(?:#{1,6}\s*|(?:Observations|Reasoning|Description|Confidence|Heads|Example|probabilities|order)\s*:)",following,re.I):break
            content.append(following)
        return " ".join(content).lower()
    return ""


def grade(job,text):
    exp=job["experiment"];case=job["case"]
    label={1:"Game",2:"Images",3:"corpus"}[exp]
    answer=section(text,label)
    keywords={"breakout":["breakout","brick"],"pong":["pong"],"pacman":["pac-man","pacman"],
        "freeway":["freeway","frogger"],"circle":["circle","disc","round","oval"],
        "square":["square","rectangle","box"],"triangle":["triangle","pyramid","wedge"],
        "python":["python"],"english":["english"],"romaji":["romaji"]}
    result={"case":case,"extracted":answer,"correct":any(k in answer for k in keywords[case]),
        "grading_version":"label-section-v2","semantic_label_review":"pending"}
    if exp==3:
        result["order_extracted"]=section(text,"order")
        result["order_correct"]=result["order_extracted"].startswith("as is")
        found=[]
        for line in text.splitlines():
            line=re.sub(r"[*_`]","",line)
            m=re.search(r"^\s*(?:\|\s*|[-*]\s+)?L(\d)H(\d)\s*[:|–—-]\s*[*_`'\"]*(previous|local|broad|copy|unclear)\b",line,re.I)
            if m and m.group(3).lower()=="copy":found.append(f"{m.group(1)}.{m.group(2)}")
        truth={"python":{"2.1","3.1","3.5"},"romaji":{"3.3"},"english":{"3.1"}}[case]
        result.update(copy_heads=sorted(set(found)),copy_precision=(len(set(found)&truth)/len(set(found)) if found else 0),
            copy_recall=len(set(found)&truth)/len(truth),copy_exact_match=set(found)==truth,
            semantic_no_forward_pass_audit="pending_trace_review")
    return result


def generate(job):
    out=ROOT/"replication/runs"/job["id"];out.mkdir(parents=True,exist_ok=True)
    if (out/"result.json").exists():return
    prompt=prompt_for(job);history=[{"role":"user","content":prompt}]
    (out/"job.json").write_text(json.dumps(job,indent=2))
    for turn in range(101 if job["experiment"]==3 else 1):
        turn_dir=out/f"turn-{turn:03d}"
        if (turn_dir/"call.json").exists() and (turn_dir/"answer.md").exists():
            text=(turn_dir/"answer.md").read_text();metadata=json.loads((turn_dir/"call.json").read_text())
        else:text,metadata=call(job["model"],job["effort"],history,turn_dir)
        if metadata.get("transport") in ("openai_api","coding_plan_responses_bridge"):
            history.extend(metadata["continuation_items"])
        else:history.append({"role":"assistant","content":text})
        blocks=re.findall(r"```python\s*\n(.*?)```",text,re.S)
        if job["experiment"]!=3 or not blocks:
            result=grade(job,text);result.update(transport=metadata["transport"],turns=turn+1)
            (out/"result.json").write_text(json.dumps(result,indent=2));return
        host,remote=connection()
        inner=[remote+"/.venv-games/bin/python","-m","wsp_repro.sandbox",
            "--job",job["id"],"--language",job["case"]]
        # Set only the remote import path; model code is stdin, never shell text.
        cached=turn_dir/"sandbox.json"
        if cached.exists():result=json.loads(cached.read_text())
        else:
            if Path(remote).resolve()==ROOT.resolve():
                response=subprocess.run(inner,input=blocks[-1],text=True,capture_output=True,timeout=650,
                    env=dict(os.environ,PYTHONPATH=str(ROOT/"replication/src")))
            else:
                cmd=["ssh","-o","BatchMode=yes",host,
                    "PYTHONPATH="+shlex.quote(remote+"/replication/src")+" "+shlex.join(inner)]
                response=subprocess.run(cmd,input=blocks[-1],text=True,capture_output=True,timeout=650)
            if response.returncode:raise RuntimeError("Remote sandbox launch failed")
            result=json.loads(response.stdout);cached.write_text(json.dumps(result,indent=2))
        history.append({"role":"user","content":result["out"]})
    raise RuntimeError("Python run budget exhausted without a final answer")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--experiment",type=int,choices=[1,2,3],required=True)
    ap.add_argument("--model",choices=["astra","deepseek"],required=True)
    args=ap.parse_args()
    for job in make_spec()["jobs"]:
        if job["experiment"]==args.experiment and job["model"]==args.model:
            generate(job);print(job["id"],flush=True)
