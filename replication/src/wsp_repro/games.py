"""Shared observation/action adapter; environment state never goes to policy code."""
import numpy as np
from .snake import Env as Snake

MAPPING={
    "breakout":(("paddle",),(),("brick",),("ball",),("trail",),()),
    "seaquest":(("sub_front","sub_back"),("enemy_fish","enemy_sub","enemy_bullet"),
        ("enemy_fish","enemy_sub","diver"),("friendly_bullet",),("trail","sub_back"),
        ("oxygen_guage","diver_guage")),
    "asterix":(("player",),("enemy",),("gold",),(),("trail",),()),
}


class Game:
    def __init__(self,name,seed,sticky=0.1):
        self.name=name;self.sticky=sticky;self.previous_action=0
        if name=="snake":self.env=Snake()
        else:
            from importlib import import_module
            self.env=import_module("minatar.environments."+name).Env()
        self.env.random=np.random.RandomState(seed)
        # Match MinAtar 1.0.15: sticky actions and world randomness share the seeded stream.
        self.sticky_rng=self.env.random
        self.env.reset()

    def observe(self):
        state=self.env.state()
        if self.name=="snake":return state.copy()
        out=np.zeros((10,10,6),bool)
        for target,names in enumerate(MAPPING[self.name]):
            for name in names:out[:,:,target]|=state[:,:,self.env.channels[name]]
        return out

    def step(self,action):
        action=int(action)
        if action not in range(6):raise ValueError("Action must be 0..5")
        if self.sticky_rng.random_sample()<self.sticky:action=self.previous_action
        self.previous_action=action
        return self.env.act(action)


def evaluate(game,policy,seeds=range(1000),max_steps=10000,sticky=0.1):
    scores=[];lengths=[];cuts=[]
    for seed in seeds:
        env=Game(game,int(seed),sticky);rng=np.random.RandomState(int(seed)+2_000_000)
        score=0;terminal=False
        for step in range(max_steps):
            output=policy(env.observe())
            if np.ndim(output)==0:action=int(output)
            else:
                v=np.asarray(output,np.float64)
                if v.shape!=(6,) or not np.isfinite(v).all():raise ValueError("Expected 6 finite logits")
                p=np.exp(v-v.max());p/=p.sum();action=rng.choice(6,p=p)
            reward,terminal=env.step(action);score+=reward
            if terminal:break
        scores.append(float(score));lengths.append(step+1);cuts.append(not terminal)
    a=np.asarray(scores)
    return {"episodes":len(scores),"mean":float(a.mean()),
        "episode_sample_variance":float(a.var(ddof=1)) if len(a)>1 else None,
        "standard_error":float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None,
        "mean_steps":float(np.mean(lengths)),"cut_fraction":float(np.mean(cuts)),
        "episode_scores":scores,"episode_lengths":lengths,"seeds":list(map(int,seeds))}
