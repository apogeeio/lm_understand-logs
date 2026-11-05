from wsp_repro.math_grading import extract,correct

def test_last_box_and_thinking_match_released_extractor():
    assert extract(r"<think>\boxed{1}</think>answer \boxed{\frac{2}{3}}") == r"\frac{2}{3}"
    assert extract(r"\boxed{1} then \boxed{") is None
    assert extract(r"\fbox{4}")=="4"
    assert extract(r"\fbox 4") is None
    assert extract("\\boxed 4$ other")=="4"

def test_symbolic_equivalence_and_missing_answer():
    assert correct(r"\frac{1}{2}","0.5")
    assert not correct(None,"0.5")
    assert not correct("3","4")
