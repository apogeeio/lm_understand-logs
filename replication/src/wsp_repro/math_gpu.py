"""One GPU worker for controlled practice-only forward passes; no editor API credentials.

The editor never mounts the worker queue, held-out data or model implementation.
Prefix caching is disabled so changed weights cannot reuse another model's activations.
"""
from .execution_scope import math_paused
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from . import math_weights

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/"local_repro/models/base"
PRACTICE=ROOT/"exp8_math/practice"
QUEUE=ROOT/"replication/artifacts/math-forward-queue"


def reload_worker_weights(worker,base,edit_dir):
    """Executed by vLLM's worker RPC; use its own HF-to-fused-weight loader."""
    import torch
    from safetensors import safe_open
    model=worker.model_runner.get_model()
    previous=getattr(worker,"_wsp_previous_edits",set())
    current={p.name[:-4]:p for p in Path(edit_dir).glob("*.npy")} if edit_dir else {}
    tensors=[]
    with safe_open(str(Path(base)/"model.safetensors"),framework="pt",device="cpu") as f:
        for name in sorted(previous|set(current)):
            value=torch.from_numpy(np.load(current[name],allow_pickle=False)) if name in current else f.get_tensor(name)
            tensors.append((name,value))
    loaded=model.load_weights(iter(tensors)) if tensors else set()
    params=dict(model.named_parameters());verified=[]
    for name,value in tensors:
        mapped=name.replace("model.language_model.","language_model.model.")
        if mapped in params and params[mapped].shape==value.shape:
            actual=params[mapped]
            if not torch.equal(actual.detach(),value.to(device=actual.device,dtype=actual.dtype)):
                raise RuntimeError("Reloaded model differs from submitted tensor: "+name)
            verified.append(name)
    worker._wsp_previous_edits=set(current)
    torch.cuda.synchronize()
    return {"updated_tensors":len(tensors),"edited":sorted(current),"loaded_parameters":sorted(loaded),
            "direct_tensor_checks":verified}


class MathWeightReloadMixin:
    """Trusted worker extension; RPC carries a method name and paths, never pickled code."""
    def wsp_reload_weights(self,base,edit_dir):
        return reload_worker_weights(self,base,edit_dir)


class ForwardWorker:
    def __init__(self):
        from vllm import LLM
        from transformers import AutoTokenizer
        self.tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
        self.context_limit=36864
        self.llm=LLM(model=str(BASE),dtype="bfloat16",max_model_len=self.context_limit,max_num_seqs=16,
            gpu_memory_utilization=0.45,enable_prefix_caching=False,enable_sleep_mode=True,
            worker_extension_cls="wsp_repro.math_gpu.MathWeightReloadMixin",
            limit_mm_per_prompt={"image":0,"video":0},seed=0)
        self.sleeping=False;self.tf=None
        self.practice={}
        for path in PRACTICE.glob("*.jsonl"):
            for line in path.open():
                row=json.loads(line);self.practice[row["id"]]=row

    def wake(self):
        if self.tf is not None:
            self.tf.cpu();self.tf=None;gc.collect()
            import torch
            torch.cuda.empty_cache()
        if self.sleeping:self.llm.wake_up();self.sleeping=False

    def sleep(self):
        if not self.sleeping:self.llm.sleep(level=1);self.sleeping=True

    def select(self,edited):
        self.wake()
        self.reload_audit=self.llm.collective_rpc("wsp_reload_weights",args=(str(BASE),str(edited) if edited else ""))

    def generate(self,request,work,edit_dir,pass_id):
        from vllm import SamplingParams
        from . import math_grading as mod
        model=request.get("model","original");n=int(request.get("n",1));cap=int(request.get("max_tokens",32768))
        ids=request.get("problems",[]);custom=request.get("prompts")
        if any(i not in self.practice for i in ids):raise ValueError("Only released practice IDs are allowed")
        if custom is not None and ids and len(custom)!=len(ids):raise ValueError("prompts/problems lengths differ")
        rows=[self.practice[i] for i in ids] if ids else [{"id":f"custom-{i}","problem":p} for i,p in enumerate(custom or [])]
        count=len(rows)*n
        if n<1 or count<1 or count>(256 if model=="both" else 512) or not 1<=cap<=32768:
            raise ValueError("Generation size/token budget exceeded")
        prompts=[];params=[]
        for i,row in enumerate(rows):
            user=custom[i] if custom is not None else row["problem"]+"\nPlease reason step by step, and put your final answer within \\boxed{}."
            prompts.append(self.tok.apply_chat_template([{"role":"user","content":user}],tokenize=False,
                add_generation_prompt=True,enable_thinking=True))
            seed=int.from_bytes(hashlib.sha256(row["id"].encode()).digest()[:4],"big")
            params.append(SamplingParams(n=n,temperature=0.6,top_p=0.95,top_k=20,presence_penalty=1.5,
                max_tokens=cap,seed=seed))
        arms=["original","edited"] if model=="both" else [model]
        if any(a not in ("original","edited") for a in arms):raise ValueError("Invalid model selector")
        results={};pair={}
        for arm in arms:
            self.select(edit_dir if arm=="edited" else None)
            outputs=self.llm.generate(prompts,params,use_tqdm=False)
            records=[];activation_inputs=[]
            for row,prompt,result in zip(rows,prompts,outputs):
                for k,response in enumerate(result.outputs):
                    if response.finish_reason not in ("stop","length"):raise RuntimeError("Incomplete model generation")
                    prediction=mod.extract(response.text)
                    good=mod.correct(prediction,str(row["answer"])) if "answer" in row else None
                    records.append({"id":row["id"],"source":row.get("source","custom"),"sample":k,"text":response.text,"pred":prediction,
                        "correct":good,"tokens":len(response.token_ids),"cut_at_max_tokens":response.finish_reason=="length"})
                    if request.get("activations"):
                        activation_inputs.append({"ids":self.tok.encode(prompt,add_special_tokens=False)+list(response.token_ids[:8192]),
                            "prompt_tokens":len(self.tok.encode(prompt,add_special_tokens=False)),"id":row["id"],"correct":good})
            name=f"gen_{pass_id}"+("_"+arm if model=="both" else "")
            (work/(name+".reload.json")).write_text(json.dumps(self.reload_audit,indent=2))
            with (work/(name+".jsonl")).open("w") as f:
                for r in records:f.write(json.dumps(r)+"\n")
            graded=[r for r in records if r["correct"] is not None]
            results[arm]={"generations":len(records),"accuracy":sum(r["correct"] for r in graded)/len(graded) if graded else None,
                "mean_tokens":sum(r["tokens"] for r in records)/len(records),
                "cut_share":sum(r["cut_at_max_tokens"] for r in records)/len(records),"file":"/work/"+name+".jsonl"}
            results[arm]["by_source"]={source:{"n":len(part),"accuracy":sum(r["correct"] for r in part)/len(part)}
                for source in sorted({r["source"] for r in graded})
                for part in [[r for r in graded if r["source"]==source]]}
            results[arm]["failures"]={"cut_at_max_tokens":sum(r["cut_at_max_tokens"] for r in records),
                "no_answer_finished":sum(not r["cut_at_max_tokens"] and r["pred"] is None for r in records),
                "wrong_answer_finished":sum(not r["cut_at_max_tokens"] and r["pred"] is not None and r["correct"] is False for r in records)}
            pair[arm]=[r["correct"] for r in records]
            if activation_inputs:
                results[arm]["activations"]=self.generation_activations(activation_inputs,request["activations"],
                    work/(name+"_acts"),edit_dir if arm=="edited" else None)
        if model=="both" and all(x is not None for x in pair["original"]+pair["edited"]):
            d=np.array(pair["edited"],float)-np.array(pair["original"],float)
            results["paired"]={"difference":float(d.mean()),"standard_error":float(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else None,
                "wrong_to_right":int((d==1).sum()),"right_to_wrong":int((d==-1).sum())}
        return results

    def load_tf(self,edit_dir,attention=False):
        import torch
        from transformers import AutoConfig,AutoModelForCausalLM
        from safetensors.torch import load_file
        self.sleep()
        if self.tf is not None:self.tf.cpu();self.tf=None;gc.collect();torch.cuda.empty_cache()
        config=AutoConfig.from_pretrained(BASE,local_files_only=True).text_config
        state={k.replace("model.language_model.","model."):v for k,v in load_file(BASE/"model.safetensors").items()
               if k.startswith("model.language_model.")}
        for p in Path(edit_dir).glob("*.npy") if edit_dir else []:
            key=p.name[:-4].replace("model.language_model.","model.")
            state[key]=torch.from_numpy(np.load(p,allow_pickle=False)).to(state[key].dtype)
        state["lm_head.weight"]=state["model.embed_tokens.weight"]
        model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,attn_implementation="eager" if attention else "sdpa")
        model.load_state_dict(state,strict=True,assign=True);model.tie_weights();del state
        self.tf=model.to("cuda").eval()
        return self.tf

    def heldout(self,request,work,edit_dir,pass_id):
        """Internal evaluator only. Editor controllers reject this request kind."""
        from vllm import SamplingParams
        from .math_grading import extract,correct
        samples={"math500":4,"gsm8k":2,"amc23":16,"minerva":4,"olympiad":4,"ifeval":2,"mmlupro1k":1,"gpqa":4}
        name=request["benchmark"]
        if name not in samples:raise ValueError("Unsupported held-out benchmark")
        path=ROOT/"replication/artifacts/bench"/(name+".jsonl")
        with path.open() as f:rows=[json.loads(l) for l in f]
        start=int(request["start"]);end=int(request["end"]);n=samples[name]
        if not 0<=start<end<=len(rows) or (end-start)*n>16:raise ValueError("Invalid held-out chunk")
        rows=rows[start:end]
        prompts=[self.tok.apply_chat_template([{"role":"user","content":r["prompt"] if r.get("kind") else r["problem"]+
            "\nPlease reason step by step, and put your final answer within \\boxed{}."}],tokenize=False,
            add_generation_prompt=True,enable_thinking=True) for r in rows]
        self.select(edit_dir)
        sampling=dict(n=n,temperature=0.6,top_p=0.95,top_k=20,presence_penalty=1.5,max_tokens=32768,seed=0)
        began=time.time();outputs=self.llm.generate(prompts,SamplingParams(**sampling),use_tqdm=False)
        records=[]
        for row,result in zip(rows,outputs):
            for k,response in enumerate(result.outputs):
                if response.finish_reason not in ("stop","length"):raise RuntimeError("Incomplete held-out generation")
                if row.get("kind")=="ifeval":
                    from .extra_grading import ifeval
                    good,pred,extra=ifeval(row,response.text)
                elif row.get("kind")=="mc":
                    from .extra_grading import mcq_extract
                    pred=mcq_extract(response.text);good=pred==row["answer"];extra={}
                else:pred=extract(response.text);good=correct(pred,row["answer"]);extra={}
                records.append({"id":row["id"],"sample":k,"text":response.text,"prediction":pred,
                    "correct":good,"tokens":len(response.token_ids),
                    "hit_token_limit":response.finish_reason=="length",**extra})
        target=work/(pass_id+".jsonl");tmp=target.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records));tmp.replace(target)
        return {"benchmark":name,"start":start,"end":end,"generations":len(records),
            "ok":sum(r["correct"] for r in records),"cut":sum(r["hit_token_limit"] for r in records),
            "tokens":sum(r["tokens"] for r in records),"sampling":sampling,"seconds":time.time()-began,
            "data_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"reload_audit":self.reload_audit,
            "context_limit":self.context_limit,"max_num_seqs":16,"gpu_memory_utilization":.45}

    def teacher(self,request,work,edit_dir,pass_id):
        import torch
        model=request.get("model","original")
        if model not in ("original","edited"):raise ValueError("Teacher forced model must be original/edited")
        ret=request.get("return",{});texts=request.get("texts")
        if texts is None:
            texts=[self.tok.apply_chat_template(c,tokenize=False,add_generation_prompt=False,
                enable_thinking=request.get("enable_thinking",False)) for c in request.get("chats",[])]
        ids=[self.tok.encode(t,add_special_tokens=False) for t in texts]
        if not 1<=len(ids)<=1024 or max(map(len,ids))>32768 or sum(map(len,ids))>524288:
            raise ValueError("Teacher-forced text/token budget exceeded")
        att=ret.get("attn_layers",[])
        if att and (len(ids)>2 or max(map(len,ids))>2048 or any(i not in [3,7,11,15,19,23] for i in att)):
            raise ValueError("Attention request exceeds scope/size")
        hidden=ret.get("hidden_states",[]);mlp=ret.get("mlp_act",[]);topk=int(ret.get("topk",0))
        if any(i not in range(25) for i in hidden) or any(i not in range(24) for i in mlp) or not 0<=topk<=50:
            raise ValueError("Invalid layer/topk request")
        folder=work/f"fwd_{pass_id}";folder.mkdir(exist_ok=True)
        ntext=len(ids);length=max(map(len,ids));shapes={}
        for layer in hidden:shapes[f"hidden_{layer}"]=(ntext,length,1024)
        for layer in mlp:shapes[f"mlp_{layer}"]=(ntext,length,3584)
        for layer in att:shapes[f"attention_{layer}"]=(ntext,8,length,length)
        if ret.get("token_logprobs"):shapes["token_logprobs"]=(ntext,length)
        if topk:shapes["topk_values"]=(ntext,length,topk);shapes["topk_ids"]=(ntext,length,topk)
        total=sum(int(np.prod(s))*(8 if k=="topk_ids" else 4) for k,s in shapes.items())
        if total>6*1024**3:raise ValueError("Requested padded output exceeds 6 GiB")
        self.load_tf(edit_dir if model=="edited" else None,bool(att))
        metadata={"lengths":list(map(len,ids)),"files":{},"padding":"right zero-padding; first token logprob is NaN"}
        maps={}
        for key,shape in shapes.items():
            maps[key]=np.lib.format.open_memmap(folder/(key+".npy"),mode="w+",shape=shape,
                dtype=np.int64 if key=="topk_ids" else np.float32)
            maps[key][:]=0
            metadata["files"][key]={"file":key+".npy","shape":list(shape)}
        for index,tokens in enumerate(ids):
            arrays=self.tf_arrays(tokens,hidden,mlp,att,ret.get("token_logprobs",False),topk)
            for name,array in arrays.items():
                if name.startswith("attention_"):maps[name][index,:,:len(tokens),:len(tokens)]=array
                elif name=="token_logprobs":
                    maps[name][index,0]=np.nan;maps[name][index,1:len(tokens)]=array
                else:maps[name][index,:len(tokens)]=array
            for mmap in maps.values():mmap.flush()
        maps.clear()
        (folder/"meta.json").write_text(json.dumps(metadata,indent=2))
        return {"folder":"/work/"+folder.name,"texts":len(ids),"tokens":sum(map(len,ids)),"meta":metadata}

    def tf_arrays(self,tokens,hidden,mlp,att,logprobs,topk):
        import torch
        captures={};handles=[]
        def capture_hidden(index,value):
            if isinstance(value,tuple):value=value[0]
            captures[f"hidden_{index}"]=value[0].detach().float().cpu().numpy()
        if 0 in hidden:
            handles.append(self.tf.model.embed_tokens.register_forward_hook(lambda m,a,o:capture_hidden(0,o)))
        for layer in hidden:
            if layer>0:
                handles.append(self.tf.model.layers[layer-1].register_forward_hook(
                    lambda m,a,o,l=layer:capture_hidden(l,o)))
        for layer in mlp:
            handles.append(self.tf.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
                lambda module,args,l=layer:captures.update({f"mlp_{l}":args[0].detach().float().cpu().numpy()[0]})))
        try:
            x=torch.tensor([tokens],device="cuda")
            with torch.inference_mode():
                # Run the backbone; projecting every 32K token to a 248K vocabulary would exhaust VRAM.
                result=self.tf.model(x,use_cache=False,output_hidden_states=False,output_attentions=bool(att))
            if att:
                for layer,array in zip([3,7,11,15,19,23],result.attentions):
                    if layer in att:captures[f"attention_{layer}"]=array[0].float().cpu().numpy()
            if logprobs or topk:
                lp=[];tv=[];ti=[]
                with torch.inference_mode():
                    for start in range(0,len(tokens),64):
                        values=self.tf.lm_head(result.last_hidden_state[0,start:start+64]).float().log_softmax(-1)
                        if logprobs:
                            end=min(start+len(values),len(tokens)-1)
                            if end>start:
                                lp.append(values[:end-start].gather(1,x[0,start+1:end+1,None]).squeeze(1).cpu().numpy())
                        if topk:
                            v,i=values.topk(topk,dim=-1);tv.append(v.cpu().numpy());ti.append(i.cpu().numpy())
                if logprobs:captures["token_logprobs"]=np.concatenate(lp) if lp else np.empty(0,np.float32)
                if topk:captures["topk_values"]=np.concatenate(tv);captures["topk_ids"]=np.concatenate(ti)
            return captures
        finally:
            for h in handles:h.remove()

    def generation_activations(self,rows,settings,folder,edit_dir):
        import torch
        folder.mkdir(exist_ok=True);self.load_tf(edit_dir)
        layers=settings.get("layers",[]);mlp=layers if settings.get("mlp") else []
        mlp=[l for l in mlp if l in range(24)]
        index=[]
        for i,row in enumerate(rows):
            arrays=self.tf_arrays(row["ids"],layers,mlp,[],False,0);record={k:v for k,v in row.items() if k!="ids"}
            record["files"]={}
            n=row["prompt_tokens"]
            for key,a in arrays.items():
                summary=np.stack([a[n-1],a[n:].mean(0) if len(a)>n else a[n-1]])
                filename=f"sample_{i}_{key}.npy";np.save(folder/filename,summary);record["files"][key]=filename
            index.append(record)
        (folder/"index.json").write_text(json.dumps(index,indent=2))
        return "/work/"+folder.name


def main():
    if math_paused(ROOT):return
    import os
    import atexit
    import signal
    QUEUE.mkdir(parents=True,exist_ok=True)
    state=QUEUE.parent/"math-gpu-state.json"
    def status(phase,**fields):
        tmp=state.with_suffix(".tmp")
        tmp.write_text(json.dumps({"phase":phase,"pid":os.getpid(),"updated":time.time(),**fields}))
        tmp.replace(state)
    status("initializing")
    worker=ForwardWorker()
    atexit.register(worker.llm.llm_engine.engine_core.shutdown)
    def terminate(signum,frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,terminate)
    worker.sleep();status("ready")
    print("WSP_MATH_GPU_READY",flush=True)
    # A previous worker may have died after claiming a request but before replying.
    # Its editor is still waiting; recover the same receipt without spending another pass.
    for processing in QUEUE.glob("*.request.processing"):
        response=processing.with_name(processing.name.split(".request.")[0]+".response.json")
        if not response.exists():processing.rename(processing.with_suffix(".json"))
    while True:
        pending=sorted(QUEUE.glob("*.request.json"),key=lambda p:(p.name.startswith("eval-"),p.stat().st_mtime))
        if not pending:time.sleep(1);continue
        path=pending[0];processing=path.with_suffix(".processing");path.rename(processing)
        envelope=json.loads(processing.read_text());start=time.time()
        status("running",request=path.name)
        fatal=False
        try:
            work=Path(envelope["work"]).resolve()
            allowed=(ROOT/"replication/runs").resolve()
            if allowed not in work.parents:raise ValueError("Invalid work root")
            edited=work/"edited";math_weights.validate(BASE,edited)
            request=envelope["request"];kind=request.get("type")
            if kind=="generate":result=worker.generate(request,work,edited,envelope["pass_id"])
            elif kind=="teacher_forced":result=worker.teacher(request,work,edited,envelope["pass_id"])
            elif kind=="heldout":result=worker.heldout(request,work,edited,envelope["pass_id"])
            else:raise ValueError("Unknown forward request type")
            response={"ok":True,"result":result,"seconds":time.time()-start}
        except Exception as e:
            response={"ok":False,"error_type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}
            fatal="EngineDead" in type(e).__name__ or "engine core is dead" in str(e).lower()
        finally:
            try:worker.sleep()
            except Exception:pass
            if worker.tf is not None:
                import torch
                worker.tf.cpu();worker.tf=None;gc.collect();torch.cuda.empty_cache()
        output=processing.with_name(processing.name.split(".request.")[0]+".response.json")
        tmp=output.with_suffix(".tmp");tmp.write_text(json.dumps(response));tmp.replace(output)
        if fatal:
            status("engine_failed",last_request=path.name)
            raise RuntimeError("GPU engine failed; response retained, restart the worker")
        status("ready",last_request=path.name,last_ok=response["ok"])


if __name__=="__main__":main()
