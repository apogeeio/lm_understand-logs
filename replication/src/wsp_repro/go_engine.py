"""CPU PGX games/positions tools; never imported or mounted in the editor sandbox."""
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
from .go_weights import load_policy,forward

ROOT=Path(__file__).resolve().parents[3]

@lru_cache(maxsize=1)
def game_runner():
    import jax
    import jax.numpy as jnp
    from pgx.go import Go
    from pgx._src.games.go import _count_scores
    if jax.default_backend()!="cpu":raise RuntimeError("Go evaluation must be CPU-only")
    env=Go(size=9,komi=7.5,history_length=1,max_terminal_steps=162)
    def nn(w,x):
        x=x.reshape(243).astype(jnp.float32)
        for i in range(1,7):x=jax.nn.relu(x@w[f"L{i}.W"]+w[f"L{i}.b"])
        return x@w["out.W"]+w["out.b"]
    def episode(us,op,index,total,seed):
        key=jax.random.fold_in(jax.random.PRNGKey(seed),index)
        black=index<total//2;order=jnp.where(black,jnp.array([0,1]),jnp.array([1,0]))
        s=env.init(key).replace(_player_order=order,current_player=order[0])
        def step(carry,_):
            state,key,final_reward,length,passes,finite=carry
            key,ak=jax.random.split(key);active=~state.terminated;our_turn=state.current_player==0
            lg=jnp.where(our_turn,nn(us,state.observation),nn(op,state.observation))
            finite=finite&jnp.isfinite(lg).all();masked=jnp.where(state.legal_action_mask,lg,-jnp.inf)
            probs=jax.nn.softmax(masked);topv,topi=jax.lax.top_k(probs,5)
            action=jax.random.categorical(ak,masked)
            nxt=env.step(state,action)
            reward=jnp.where(active&nxt.terminated,nxt.rewards[0],final_reward)
            length+=active;passes+=active&our_turn&(action==81)
            board=jnp.sign(nxt._x.board).astype(jnp.int8).reshape(9,9)
            record=(jnp.where(active,action,-1),board,state.legal_action_mask,topi,topv,active&our_turn)
            return (nxt,key,reward,length,passes,finite),record
        carry,records=jax.lax.scan(step,(s,key,jnp.float32(0),jnp.int32(0),jnp.int32(0),jnp.bool_(True)),None,length=162)
        state,key,reward,length,passes,finite=carry
        scores=_count_scores(state._x,9);margin=(scores[0]-scores[1]-7.5)*jnp.where(black,1,-1)
        captures=state._x.num_captured[jnp.where(black,0,1)];lost=state._x.num_captured[jnp.where(black,1,0)]
        ending=jnp.where(state._x.is_psk,0,jnp.where(state._x.consecutive_pass_count>=2,1,2))
        return (reward,length,passes,margin,captures,lost,ending,finite),records
    return jax.jit(jax.vmap(episode,in_axes=(None,None,0,None,None)))

def games(weights,opponent,n,seed,out,save_logs=True):
    import jax.numpy as jnp
    if n<2 or n%2:raise ValueError("Go game count must be positive and even")
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    us={k:jnp.asarray(v) for k,v in weights.items()};op={k:jnp.asarray(v) for k,v in opponent.items()}
    all_games=[];boards=[];valid=[];started=time.time();runner=game_runner()
    for start in range(0,n,16):
        ids=np.arange(start,start+16,dtype=np.int32)
        metrics,records=runner(us,op,jnp.asarray(ids),n,seed)
        metrics=[np.asarray(a) for a in metrics];records=[np.asarray(a) for a in records]
        for i,index in enumerate(ids[:min(16,n-start)]):
            reward,length,passes,margin,captured,lost,ending,finite=[a[i] for a in metrics]
            if not finite:raise ValueError("Nonfinite Go policy logits")
            row={"id":int(index),"black":bool(index<n//2),"win":bool(reward>0),"steps":int(length),
                "passes":int(passes),"area_margin":float(margin),"captured":int(captured),"lost":int(lost),
                "ending":["repetition","two_passes","move_cap"][int(ending)]}
            row.update(net_color="black" if row["black"] else "white",result=1 if row["win"] else -1,
                margin=row["area_margin"],end=row["ending"],stones_captured_by_net=row["captured"],stones_lost_by_net=row["lost"])
            if save_logs:
                acts,b,v,topi,topv,ours=[a[i] for a in records]
                row["moves"]=[{"ply":t,"by":"net" if ours[t] else "opponent","move":int(acts[t]),
                    **({"top5":[[int(a),float(p)] for a,p in zip(topi[t],topv[t])]} if ours[t] else {})}
                    for t in range(int(length))]
                boards.append(b.reshape(162,81));valid.append(np.arange(162)<int(length))
            all_games.append(row)
    (folder/"games.json").write_text(json.dumps(all_games))
    if save_logs:np.save(folder/"boards.npy",np.asarray(boards));np.save(folder/"valid.npy",np.asarray(valid))
    summary={"games":n,"wins":sum(r["win"] for r in all_games),"score":sum(r["win"] for r in all_games)/n,
        "score_as_black":np.mean([r["win"] for r in all_games if r["black"]]).item(),
        "score_as_white":np.mean([r["win"] for r in all_games if not r["black"]]).item(),
        **{"mean_"+k:np.mean([r[k] for r in all_games]).item() for k in ["steps","passes","area_margin","captured","lost"]},
        "endings":{e:sum(r["ending"]==e for r in all_games) for e in ["repetition","two_passes","move_cap"]},
        "seed":seed,"seconds":time.time()-started,"runtime":"PGX2.6.0 Go9x9, komi7.5,162moves,3planes,CPU"}
    summary.update(losses=n-summary["wins"],mean_margin=summary["mean_area_margin"],mean_length=summary["mean_steps"],
        stones_captured_per_game=summary["mean_captured"],stones_lost_per_game=summary["mean_lost"],passes_per_game=summary["mean_passes"])
    (folder/"summary.json").write_text(json.dumps(summary,indent=2));return summary

def positions(weights,request,out):
    import jax
    from pgx.go import Go
    items=request.get("positions",[])
    if not 1<=len(items)<=256:raise ValueError("Positions pass permits 1..256 positions")
    env=Go(size=9,komi=7.5,history_length=1,max_terminal_steps=162);inputs=[];masks=[]
    for item in items:
        if "moves" in item:
            state=env.init(jax.random.PRNGKey(0))
            for move in item["moves"]:
                action=81 if move=="pass" else int(move)
                if state.terminated or not 0<=action<82 or not bool(state.legal_action_mask[action]):raise ValueError("Illegal move history")
                state=env.step(state,action)
            x=np.asarray(state.observation,np.float32);mask=np.asarray(state.legal_action_mask,bool)
        else:
            board=item["board"]
            if len(board)!=9 or any(len(r)!=9 or set(r)-set("XO.") for r in board):raise ValueError("Invalid board")
            if item.get("to_move") not in ("black","white"):raise ValueError("Invalid colour")
            b=np.array([list(r) for r in board]);mine="X" if item["to_move"]=="black" else "O"
            x=np.stack([b==mine,b==("O" if mine=="X" else "X"),np.full((9,9),mine=="O")],axis=-1).astype(np.float32)
            mask=np.r_[b.reshape(-1)==".",True]
        inputs.append(x);masks.append(mask)
    x=np.asarray(inputs);mask=np.asarray(masks);lg,hidden=forward(weights,x,True)
    masked=np.where(mask,lg,-np.inf);p=np.exp(masked-masked.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    for name,a in [("inputs",x.reshape(len(items),243)),("legal",mask),("logits",lg),("probabilities",p)]:np.save(folder/(name+".npy"),a)
    if request.get("activations"):
        for i,a in enumerate(hidden,1):np.save(folder/f"hidden_{i}.npy",a)
    result=[]
    for i in range(len(items)):
        top=np.argsort(-p[i])[:3];result.append({"moves":top.tolist(),"probabilities":p[i,top].tolist(),"logits":lg[i,top].tolist()})
    (folder/"summary.json").write_text(json.dumps(result,indent=2));return result
