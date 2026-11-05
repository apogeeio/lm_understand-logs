from pathlib import Path
import pytest
from wsp_repro.run_reading import grade

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("language",["python","romaji","english"])
def test_grade_released_language_answers(language):
    text=(ROOT/"exp3_language/runs"/(language+"_r0")/"final_answer.md").read_text()
    result=grade({"experiment":3,"case":language},text)
    assert result["correct"] and result["copy_exact_match"]


def test_copy_in_evidence_is_not_a_copy_label():
    text="corpus: english\nL3H1: copy — matching mechanism\nL3H3: unclear — no evidence of a copy circuit"
    assert grade({"experiment":3,"case":"english"},text)["copy_heads"]==["3.1"]
def test_markdown_answer_sections_are_read_without_using_reasoning_keywords():
    from wsp_repro.run_reading import grade
    assert grade({"experiment":1,"case":"breakout"},"**Game:** **Breakout**")["correct"]
    assert grade({"experiment":1,"case":"pacman"},"## Game\n\n**A Pac-Man style game.**\n\n## Description\nother")["correct"]
    assert grade({"experiment":2,"case":"circle"},"**Images:**\nOne hollow circle per image.")["correct"]
    assert not grade({"experiment":1,"case":"breakout"},"Reasoning: Breakout is unlikely.\n\n**Game:** Snake")["correct"]
