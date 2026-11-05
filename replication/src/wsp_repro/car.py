"""Rounded actor from the actual prompt, safe edit parsing, and Kinetix evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from .edits import apply
from .providers import call
from .spec import make_spec

ROOT=Path(__file__).resolve().parents[3]
PROMPT=ROOT/"exp4_fix_car/runs/oneshot_repair/reply/prompt.txt"

def rounded_actor():
    parts=re.split(r"^((?:embed|actor[0-4]|actor_head)\.[Wb]) \(([^)]+)\), scale ([^:]+):[^\n]*\n",
        PROMPT.read_text(),flags=re.M);out={}
    for i in range(1,len(parts),4):
        name,shape,scale,body=parts[i:i+4]
        values="\n".join(re.sub(r"^\d+:\s*","",line) for line in body.splitlines() if line.strip())
        out[name]=np.fromstring(values,sep=" ").reshape(tuple(map(int,shape.split(" x "))))*float(scale)
    assert len(out)==14 and sum(a.size for a in out.values())==171664
    return out

def generate(job):
    out=ROOT/"replication/runs"/job["id"];out.mkdir(parents=True,exist_ok=True)
    if (out/"artifact.json").exists():return
    (out/"job.json").write_text(json.dumps(job,indent=2))
    d=out/"call"
    if (d/"call.json").exists() and (d/"answer.md").exists():
        text=(d/"answer.md").read_text();meta=json.loads((d/"call.json").read_text())
    else:text,meta=call(job["model"],job["effort"],[{"role":"user","content":PROMPT.read_text()}],d)
    base=rounded_actor();weights,report=apply(text,base)
    artifact=out/"actor.npz";np.savez(artifact,**weights)
    record={"job":job,"transport":meta["transport"],"parse":report,"frozen_before_evaluation":True,
        "sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),"artifact":artifact.name,
        "changed":sum(int((weights[k]!=base[k]).sum()) for k in base)}
    (out/"artifact.json").write_text(json.dumps(record,indent=2))

def evaluate(actor,episodes,seed,output):
    import jax
    import jax.numpy as jnp
    if jax.default_backend()!="cpu":raise RuntimeError("Run car evaluation in the CPU-only simulator environment")
    from kinetix.util.saving import load_from_json_file
    from kinetix.environment.env import make_kinetix_env
    from kinetix.environment.spaces import ActionType,ObservationType
    from kinetix.environment.env_state import EnvParams
    state,static,_=load_from_json_file("l/h2_car_ramp")
    static=static.replace(frame_skip=2)
    params=EnvParams(max_timesteps=256,dense_reward_scale=0.2)
    env=make_kinetix_env(ActionType.MULTI_DISCRETE,ObservationType.SYMBOLIC_FLAT,
        lambda key:state,params,static,auto_reset=False)
    obs=env.get_obs(state)
    assert obs.shape==(679,),obs.shape
    assert np.allclose(np.asarray(state.polygon.position[5]),[2.8,.725],atol=.001)
    weights={k:jnp.asarray(v,dtype=jnp.float32) for k,v in actor.items()}
    sizes=[3,3,3,3,2,2];offsets=np.cumsum([0]+sizes)
    def logits(obs):
        x=jnp.clip(jnp.nan_to_num(obs,nan=0.,posinf=0.,neginf=0.),-10,10)
        x=jax.nn.relu(x@weights["embed.W"]+weights["embed.b"])
        for i in range(5):x=jnp.tanh(x@weights[f"actor{i}.W"]+weights[f"actor{i}.b"])
        return x@weights["actor_head.W"]+weights["actor_head.b"]
    def episode(key):
        def step(carry,_):
            s,done,win,red,invalid,ret,length,key=carry
            key,ak,sk=jax.random.split(key,3);keys=jax.random.split(ak,6)
            lg=logits(env.get_obs(s))
            action=jnp.stack([jax.random.categorical(keys[i],lg[offsets[i]:offsets[i+1]]) for i in range(6)])
            _,nxt,reward,ended,info=env.step(sk,s,action,params)
            dense=jnp.where(s.last_distance==-1,0.,-(info["distance"]-s.last_distance)*.2)
            raw_reward=reward-jnp.nan_to_num(dense)
            bad=jax.tree.reduce(jnp.logical_or,jax.tree.map(lambda x:~jnp.isfinite(x).all(),nxt),False)
            active=~done
            carry=(nxt,done|ended,win|(active&info["GoalR"]&~bad),
                red|(active&ended&(raw_reward<-.5)&~bad),invalid|(active&bad),
                ret+jnp.where(active,reward,0),length+active,key)
            return carry,None
        initial=(state,jnp.array(False),jnp.array(False),jnp.array(False),jnp.array(False),jnp.float32(0),jnp.int32(0),key)
        result,_=jax.lax.scan(step,initial,None,length=256)
        return result[2:7]
    # Compile a bounded batch; serial batches cap CPU/RAM without changing policy or episode length.
    batch=32;runner=jax.jit(jax.vmap(episode));records=[];started=time.time()
    for start in range(0,episodes,batch):
        ids=np.arange(start,min(start+batch,episodes))
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(seed),i))(jnp.asarray(ids))
        values=runner(keys)
        for values_i in zip(*map(np.asarray,values)):
            win,red,invalid,ret,length=values_i
            records.append({"success":bool(win),"red":bool(red),"invalid":bool(invalid),
                "return":float(ret),"steps":int(length)})
    report={"episodes":episodes,"seed":seed,"success":sum(r["success"] for r in records)/episodes,
        "red":sum(r["red"] for r in records),"invalid":sum(r["invalid"] for r in records),
        "timeout":sum(not(r["success"] or r["red"] or r["invalid"]) for r in records),
        "mean_return":float(np.mean([r["return"] for r in records])),"seconds":time.time()-started,
        "runtime":"Kinetix3.0.2+jax2d1.0.1; CPU FP32 actor","frame_skip":2,"max_steps":256,
        "dense_reward_scale":.2,"records":records}
    Path(output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!="records"}),flush=True)
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--generate",choices=["astra","deepseek"])
    ap.add_argument("--actor");ap.add_argument("--episodes",type=int,default=1024)
    ap.add_argument("--seed",type=int,default=1401);ap.add_argument("--output")
    args=ap.parse_args()
    if args.generate:
        job=next(j for j in make_spec()["jobs"] if j["experiment"]==4 and j["model"]==args.generate)
        while True:
            try:generate(job);break
            except Exception as e:
                print(json.dumps({"error":type(e).__name__,"message":str(e)}),flush=True)
                if type(e).__name__ in {"RateLimited","TransientProviderError","QuotaExhausted","URLError","TimeoutError"}:
                    time.sleep(1800);continue
                raise
    else:
        if not args.output:ap.error("--output required")
        actor=dict(np.load(args.actor)) if args.actor else rounded_actor()
        evaluate(actor,args.episodes,args.seed,args.output)

if __name__=="__main__":main()
