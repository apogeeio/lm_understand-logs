"""Snake extension, not an upstream MinAtar game; MinAtar-compatible 10x10x6 API.

Channels: head, body (without head), food, empty, previous head, tail.
Actions: no-op/left/up/right/down/fire; no-op and fire continue; reversal is ignored.
Reward +1 per food, no survival reward. No wrapping. Moving into a vacating tail is legal.
Sticky action is applied by the common wrapper, not here.
"""
import numpy as np


class Env:
    channels={"head":0,"body":1,"food":2,"unused":3,"trail":4,"tail":5}
    action_map=["n","l","u","r","d","f"]

    def __init__(self,ramping=None):
        self.random=np.random.RandomState();self.reset()

    def reset(self):
        self.body=[(5,5),(5,4),(5,3)]
        self.direction=(0,1);self.previous=(5,4);self.terminal=False
        self._food()

    def _food(self):
        empty=[(r,c) for r in range(10) for c in range(10) if (r,c) not in self.body]
        self.food=empty[self.random.randint(len(empty))] if empty else None

    def act(self,a):
        if self.terminal:return 0,True
        directions={1:(0,-1),2:(-1,0),3:(0,1),4:(1,0)}
        direction=directions.get(int(a),self.direction)
        if direction==(-self.direction[0],-self.direction[1]):direction=self.direction
        self.direction=direction
        r,c=self.body[0];nxt=(r+direction[0],c+direction[1]);eating=nxt==self.food
        occupied=self.body if eating else self.body[:-1]
        if not (0<=nxt[0]<10 and 0<=nxt[1]<10) or nxt in occupied:
            self.terminal=True;return 0,True
        self.previous=self.body[0];self.body.insert(0,nxt)
        if eating:
            if len(self.body)==100:self.terminal=True;self.food=None
            else:self._food()
            return 1,self.terminal
        self.body.pop();return 0,False

    def state(self):
        a=np.zeros((10,10,6),dtype=bool)
        r,c=self.body[0];a[r,c,0]=True
        for r,c in self.body[1:]:a[r,c,1]=True
        if self.food is not None:
            r,c=self.food;a[r,c,2]=True
        r,c=self.previous;a[r,c,4]=True
        r,c=self.body[-1];a[r,c,5]=True
        return a

    def state_shape(self):return [10,10,6]
    def minimal_action_set(self):return [1,2,3,4]
    def difficulty_ramp(self):return None
