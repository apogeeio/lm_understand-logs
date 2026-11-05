"""Derive prompts from upstream; scaling changes all dimensions and numeric examples."""
from pathlib import Path
import inspect
import re
from .spec import parameter_count
from . import snake

ROOT=Path(__file__).resolve().parents[3]


def source_prompt(game):
    if game!="snake":
        return (ROOT/"exp5_write_from_scratch/runs"/game/"message.txt").read_text()
    base=(ROOT/"exp5_write_from_scratch/runs/breakout/message.txt").read_text()
    first=base[:base.index("THE GAME:")]
    network=base[base.index("THE NETWORK"):]
    description='''THE GAME: Snake, a new MinAtar-compatible extension on a 10 x 10 grid.
The snake starts with length 3, heading right. Food is uniform over empty cells.
Eating gives +1 and grows the snake; there is no survival reward. Walls do not wrap.
Hitting a wall or body ends the game, but entering the tail cell is legal when it vacates.
Immediate reversal is ignored. Filling all 100 cells wins. Episodes stop after 10,000 steps.
Actions 0 no-op, 1 left, 2 up, 3 right, 4 down, 5 fire; no-op/fire continue heading.
Sticky action probability 0.1 repeats the previously executed action.
Exact game source (the shared wrapper applies sticky actions separately):
```python
'''+inspect.getsource(snake)+'''
```
THE INPUT: 10 x 10 x 6 binary observation, flattened with i=(row*10+column)*6+channel.
channel 0=head; channel 1=body excluding head; channel 2=food; channel 3=empty;
channel 4=previous head/neck; channel 5=tail. Rows run top-to-bottom, columns left-to-right.
THE OUTPUT: six logits for no-op, left, up, right, down, fire; sampled from softmax.

'''
    return first+description+network


def make_prompt(game,kind="network",width=32):
    original=source_prompt(game)
    if kind=="python":
        observation=original[original.index("THE GAME:"):original.index("THE NETWORK")]
        return '''YOUR TASK: write a short Python controller for the game below, in one response.
No tools, code execution, simulator feedback, training or retries are available.
First explain the design briefly, then give one final ```python block defining act(obs).
obs is ONLY the current 10x10x6 binary observation described below; return an int action 0..5.
At most 40 nonblank, noncomment physical lines, including imports. You may import numpy,
math, collections and heapq. No files, network, environment access, persistent state, globals,
future simulator access or recursive calls. The same source/rules are given to the network writer.
Code is compiled once but each act call must be stateless and use only obs. No online LLM calls.
The frozen controller is evaluated on 1000 episodes, capped at 10000 steps, sticky actions 0.1.

'''+observation
    if width==32:return original
    header,tail=original.split("THE NETWORK",1)
    # Keep the original prompt, edit language and examples; modify dimensions only.
    tail=re.sub(r"\b32\b",str(width),tail).replace("20,486",f"{parameter_count(width):,}")
    tail=tail.replace("the same architecture as the reinforcement-learning networks it will be compared with",
                      "a width-scaled variant of the original architecture")
    tail=tail.replace("It is compared with networks of the same architecture trained by reinforcement learning on this game.",
                      "This is a width-scaling extension; no retrained PPO comparison is implied.")
    return header+"THE NETWORK"+tail
