"""Protocol-level checks against real released artifacts and the official environment."""
from pathlib import Path
import numpy as np
import pytest
from wsp_repro.edits import apply,KEYS
from wsp_repro.games import Game
from wsp_repro.snake import Env
from wsp_repro.prompts import make_prompt
from wsp_repro.code_policy import extract
from wsp_repro.spec import make_spec

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("game",["breakout","seaquest","asterix"])
def test_published_reply_rebuilds_bit_exact_weights(game):
    text=(ROOT/"exp5_write_from_scratch/runs"/game/"reply/answer.md").read_text()
    got,report=apply(text)
    expected=np.load(ROOT/"exp5_write_from_scratch/networks"/game/"hand.npz")
    assert report["errors"]==[]
    for name,key in KEYS.items():np.testing.assert_array_equal(got[name],expected[key])


@pytest.mark.parametrize("game",["breakout","seaquest","asterix"])
def test_adapter_matches_official_physics_and_sticky_stream(game):
    from minatar import Environment
    official=Environment(game,sticky_action_prob=0.1);official.seed(17);official.reset()
    ours=Game(game,17)
    rng=np.random.RandomState(82)
    for _ in range(1000):
        a=rng.randint(6)
        assert ours.step(a)==official.act(a)
        np.testing.assert_array_equal(ours.env.state(),official.state())
        if official.env.terminal:break


def test_snake_growth_tail_vacates_and_wall_collision():
    s=Env();s.food=(5,6)
    assert s.act(3)==(1,False)
    assert len(s.body)==4 and s.body[0]==(5,6)
    s.body=[(1,1),(1,2),(2,2),(2,1)];s.direction=(0,-1);s.food=(9,9)
    assert s.act(4)==(0,False)  # enter old tail, which leaves on this step
    s.body=[(0,0),(0,1),(0,2)];s.direction=(0,-1)
    assert s.act(1)==(0,True)


def test_snake_food_and_observation_determinism():
    a=Game("snake",27);b=Game("snake",27)
    np.testing.assert_array_equal(a.observe(),b.observe())
    assert a.env.food not in a.env.body
    x=a.observe();assert x.shape==(10,10,6) and x.dtype==bool
    x[:]=False;assert a.observe().any()


def test_scaling_does_not_modify_the_game_source():
    for game in ["breakout","seaquest","asterix","snake"]:
        base=make_prompt(game)
        for w in [16,8,4]:
            prompt=make_prompt(game,width=w)
            assert prompt.split("THE NETWORK")[0]==base.split("THE NETWORK")[0]
            tail=prompt.split("THE NETWORK")[1]
            assert f"600 x {w}" in tail and "32 units" not in tail


def test_code_rejects_side_effects_and_long_code():
    with pytest.raises(ValueError):extract("```python\nimport os\ndef act(obs): return os.getpid()\n```")
    with pytest.raises(ValueError):extract("```python\ndef act(obs, x=[]): return 0\n```")
    code,report=extract("```python\nimport numpy as np\ndef act(obs):\n    return 1 if np.any(obs[:,:,0]) else 3\n```")
    assert report["physical_code_lines"]==3


def test_run_matrix_uses_independent_generations():
    jobs=make_spec()["jobs"]
    assert len(jobs)==168 and len({j["id"] for j in jobs})==168
    assert all(j["experiment"]!=7 for j in jobs)
    assert sum(j["experiment"]==5 for j in jobs)==120
    assert sum(j["experiment"]==8 for j in jobs)==18
    assert all(j["target_model"]=="Qwen/Qwen3.5-0.8B" for j in jobs if j["experiment"]==8)
