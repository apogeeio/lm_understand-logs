from wsp_repro.extra_grading import mcq_extract

def test_official_mcq_priority_and_reasoning_removal():
    assert mcq_extract("<think>The answer is (B)</think>The answer is (D)")=="D"
    assert mcq_extract("Answer: C")=="C"
    assert mcq_extract("Final result: (J)")=="J"
    assert mcq_extract("No option given.") is None
