"""Explicit stateless API calls, plus audited tool-free Codex-plan quota fallback.

Credentials never enter prompts, subprocess arguments, or the game sandbox.
No SDK sessions, previous_response_id, implicit history, tools, retrieval or memory.
"""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.request
import urllib.error

SECRET_FILE=Path.home()/".config/wsp-replication/providers.env"
SYSTEM="Answer the user's request."


class MissingCredential(RuntimeError):pass
class QuotaExhausted(RuntimeError):pass
class RateLimited(RuntimeError):pass
class TransientProviderError(RuntimeError):pass
class IncompleteModelOutput(RuntimeError):pass


def credentials(path=SECRET_FILE):
    keys={}
    if path.exists():
        if path.stat().st_mode & 0o077:raise PermissionError("Credential file must have mode 600")
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):continue
            k,sep,v=line.partition("=")
            if sep:keys[k.strip()]=v.strip().strip("\"'")
    for k in ("OPENAI_API_KEY","DEEPSEEK_API_KEY"):
        if os.environ.get(k):keys[k]=os.environ[k]
    return keys


def credential_status():
    c=credentials()
    return {k:bool(c.get(k)) for k in ("OPENAI_API_KEY","DEEPSEEK_API_KEY")}


def _post(url,payload,key):
    # File-lock slots coordinate independent processes without sharing contexts.
    folder=SECRET_FILE.parent/"locks";folder.mkdir(parents=True,exist_ok=True)
    tag="deepseek" if "api.deepseek.com" in url else "astra"
    slots=int(os.environ.get("WSP_"+tag.upper()+"_CONCURRENCY","1"))
    if not 1 <= slots <= 64:raise ValueError("Provider concurrency must be 1..64")
    while True:
        for slot in range(slots):
            name=tag+("" if slot==0 else "-"+str(slot))+".lock"
            with (folder/name).open("a") as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                return _post_unlocked(url,payload,key)
        time.sleep(.25)


def _post_unlocked(url,payload,key):
    body=json.dumps(payload).encode()
    req=urllib.request.Request(url,body,{"Authorization":"Bearer "+key,"Content-Type":"application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req,timeout=7200 if payload.get("max_tokens",0)>128000 else 1800) as response:return json.load(response)
        except urllib.error.HTTPError as e:
            try:err=json.loads(e.read()).get("error",{});code=err.get("code",err.get("type","unknown"))
            except (ValueError,AttributeError):code="unknown"
            if code in ("insufficient_quota","billing_hard_limit_reached","billing_not_active"):
                raise QuotaExhausted(f"Provider returned quota error {e.code}/{code}") from None
            if e.code==402 or code=="usage_limit_reached":
                raise QuotaExhausted(f"Provider returned quota error {e.code}/{code}") from None
            if e.code in (429,500,502,503,504) and attempt<3:
                time.sleep(2**attempt);continue
            if e.code==429:raise RateLimited(f"HTTP 429, provider error code {code}") from None
            if e.code in (500,502,503,504):raise TransientProviderError(f"HTTP {e.code}") from None
            # Do not echo provider bodies or headers: they can contain account data.
            raise RuntimeError(f"HTTP {e.code}, provider error code {code}") from None


def recorded_post(url,payload,key,out_dir):
    """Persist a returned response before validating it; a truncated output is never resampled."""
    request=out_dir/'request.json';saved=out_dir/'provider-response.json'
    if saved.exists():
        if json.loads(request.read_text())!=payload:
            raise RuntimeError('Saved provider response belongs to a different explicit request; audit required')
        return json.loads(saved.read_text())
    request.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    response=_post(url,payload,key)
    tmp=saved.with_suffix('.tmp');tmp.write_text(json.dumps(response,ensure_ascii=False));tmp.replace(saved)
    return response


def incomplete(response,payload,out_dir,model,transport,effort,reason,text):
    record={'model_group':model,'resolved_model':response.get('model'),'transport':transport,'effort':effort,
        'usage':response.get('usage',{}),'terminal_model_output':True,'reason':reason,
        'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
        'response_sha256':hashlib.sha256((out_dir/'provider-response.json').read_bytes()).hexdigest(),
        'updated':time.time()}
    (out_dir/'partial-answer.md').write_text(text)
    (out_dir/'call-failure.json').write_text(json.dumps(record,indent=2))
    raise IncompleteModelOutput(reason)


def call(model,effort,messages,out_dir,transport=None,max_output_tokens=128000):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    keys=credentials();start=time.monotonic()
    if model=="astra":
        from .settings import ROOT as SETTINGS_ROOT
        path=SETTINGS_ROOT/"configs/local.json"
        config=json.loads(path.read_text()) if path.exists() else {}
        transport=transport or "coding_plan_responses_bridge"
        bridge=transport=="coding_plan_responses_bridge"
        key="coding-plan-local-bridge" if bridge else keys.get("OPENAI_API_KEY")
        if not key:raise MissingCredential("OPENAI_API_KEY is not configured")
        payload={"model":"gpt-6-astra","instructions":SYSTEM,"input":messages,"store":False,
                 "tools":[],"reasoning":{"effort":effort,"context":"all_turns"},"max_output_tokens":128000}
        if bridge:
            url=config.get("coding_plan_base_url", "http://127.0.0.1:23880/v1").rstrip("/")+"/responses"
            # Use the configured native Responses bridge; no CLI developer/tool envelope is injected.
            payload["reasoning"]={"effort":effort,"summary":"detailed"}
            payload["include"]=["reasoning.encrypted_content"]
        else:url="https://api.openai.com/v1/responses"
        try:response=recorded_post(url,payload,key,out_dir)
        except QuotaExhausted as e:
            if bridge:raise
            (out_dir/"fallback.json").write_text(json.dumps({"cause":str(e),"from":"openai_api","to":"codex_plan"}))
            # Only single-response runs can fall back in-place. Multi-turn callers restart the run.
            if len(messages)!=1:raise
            return codex_plan(effort,messages,out_dir)
        text="\n".join(p.get("text","") for x in response.get("output",[])
                       if x.get("type")=="message" for p in x.get("content",[]) if p.get("type")=="output_text")
        transport="coding_plan_responses_bridge" if bridge else "openai_api"
        if response.get("status") not in (None,"completed"):
            incomplete(response,payload,out_dir,model,transport,effort,
                'OpenAI response incomplete: '+str(response.get('incomplete_details')),text)
        resolved=response.get("model");usage=response.get("usage",{})
    elif model=="deepseek":
        key=keys.get("DEEPSEEK_API_KEY")
        if not key:raise MissingCredential("DEEPSEEK_API_KEY is not configured")
        payload={"model":"deepseek-flash","messages":[{"role":"system","content":SYSTEM}]+messages,
                 "thinking":{"type":"enabled"},"reasoning_effort":"max","max_tokens":max_output_tokens}
        if not 1 <= max_output_tokens <= 393216:raise ValueError("DeepSeek output limit must be 1..393216")
        response=recorded_post("https://api.deepseek.com/chat/completions",payload,key,out_dir)
        choice=response["choices"][0]
        text=choice["message"].get("content") or ""
        transport="deepseek_api";resolved=response.get("model");usage=response.get("usage",{})
        if choice.get("finish_reason")!="stop":
            incomplete(response,payload,out_dir,model,transport,effort,
                'DeepSeek response ended: '+str(choice.get('finish_reason')),text)
    else:raise ValueError(model)
    metadata={"model_group":model,"resolved_model":resolved,"transport":transport,"effort":effort,
              "seconds":time.monotonic()-start,"usage":usage,"tool_free":True,
              "request_sha256":hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
    if transport in ("openai_api","coding_plan_responses_bridge"):
        # Replay only this run's output, including opaque encrypted reasoning, in stateless mode.
        metadata["continuation_items"]=response.get("output",[])
    if transport=="coding_plan_responses_bridge":
        metadata["native_output_limit"]="Responses bridge strips max_output_tokens; provider-native limit applies"
        metadata["context_envelope"]="explicit Responses payload, neutral instructions, no tools or CLI context"
    (out_dir/"request.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    (out_dir/"answer.md").write_text(text)
    (out_dir/"call.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    return text,metadata


def codex_plan(effort,messages,out_dir):
    """Best-effort API-clean CLI context; the runtime envelope is not API-identical."""
    out_dir=Path(out_dir).resolve()
    binary=shutil.which("codex")
    if not binary:raise RuntimeError("Codex CLI not installed for quota fallback")
    if len(messages)!=1:raise ValueError("Multi-turn fallback must restart in a fresh run")
    with tempfile.TemporaryDirectory(prefix="wsp-clean-") as tmp:
        work=Path(tmp);instruction=work/"neutral.md";instruction.write_text(SYSTEM+"\n")
        cmd=[binary,"exec","--ignore-user-config","--ephemeral","--skip-git-repo-check","--json",
             "-C",str(work),"-m","gpt-6-astra","-c",f'model_reasoning_effort="{effort}"',
             "-c",f'model_instructions_file={json.dumps(str(instruction))}',
             "-c","project_doc_max_bytes=0","-c",'web_search="disabled"',"-c",'sandbox_mode="read-only"',
             "-c",'approval_policy="never"',"-c","agents.enabled=false"]
        for feature in ("code_mode","shell_tool","unified_exec","shell_snapshot","view_image","multi_agent",
                        "apps","plugins","remote_plugin","browser_use","computer_use","skill_search","hooks"):
            cmd += ["-c",f"features.{feature}=false"]
        cmd += ["-o",str(out_dir/"answer.md"),"-"]
        start=time.monotonic()
        # Keep authentication only in the native CLI; no API keys are inherited by the subject process.
        env={k:v for k,v in os.environ.items() if k not in ("OPENAI_API_KEY","DEEPSEEK_API_KEY")}
        with (out_dir/"events.jsonl").open("w") as stdout,(out_dir/"stderr.log").open("w") as stderr:
            proc=subprocess.run(cmd,input=messages[0]["content"],text=True,stdout=stdout,stderr=stderr,env=env,timeout=3600)
        if proc.returncode:raise RuntimeError(f"Codex plan exited {proc.returncode}; inspect stderr.log")
    events=[json.loads(l) for l in (out_dir/"events.jsonl").read_text().splitlines() if l.strip()]
    allowed={"agent_message","reasoning"}
    tools=[e for e in events if e.get("item",{}).get("type") not in allowed and e.get("item")]
    if tools:raise RuntimeError("Tool-free audit failed; run is excluded")
    usage=[e.get("usage") for e in events if e.get("type")=="turn.completed"]
    metadata={"model_group":"astra","requested_model":"gpt-6-astra","resolved_model":None,
              "transport":"codex_plan","effort":effort,"seconds":time.monotonic()-start,
              "usage":usage,"tool_free":True,"api_identical_context":False,
              "isolation":{"fresh_ephemeral_session":True,"user_config":False,"project_docs":False,
                           "native_tools":False,"empty_working_directory":True}}
    (out_dir/"call.json").write_text(json.dumps(metadata,indent=2))
    return (out_dir/"answer.md").read_text(),metadata
