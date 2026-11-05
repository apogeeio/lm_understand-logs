import numpy as np
from wsp_repro.car import rounded_actor,ROOT
from wsp_repro.edits import apply

def test_released_car_edits_reconstruct_every_float64_value():
    base=rounded_actor()
    text=(ROOT/"exp4_fix_car/runs/oneshot_repair/reply/answer.md").read_text()
    edited,report=apply(text,base)
    assert report["applied"]==51 and not report["errors"]
    with np.load(ROOT/"exp4_fix_car/networks/edited_actor.npz") as released:
        for name in edited:assert np.array_equal(edited[name],released[name]),name
    assert sum((edited[n]!=base[n]).sum() for n in base)==6073
