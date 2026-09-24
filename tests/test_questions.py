import pytest

from reflexguard.modes import Mode
from reflexguard.questions import SCORE_LEVELS, build_questions, label, mode_score

CATS = {"pii_exposure": "names, emails", "violence": ""}


def test_label():
    assert label("pii_exposure", "names, emails") == "pii exposure (names, emails)"
    assert label("violence") == "violence"


def test_noul_questions():
    qs = build_questions(CATS, [Mode.NOUL])
    assert qs == {
        "noul:pii_exposure": {"type": "noul", "instructions": "Does `text` involve pii exposure (names, emails)?"},
        "noul:violence": {"type": "noul", "instructions": "Does `text` involve violence?"},
    }


def test_choice_question_is_binary_per_category():
    q = build_questions(CATS, [Mode.CHOICE])["choice:violence"]
    assert q["type"] == "choice"
    assert q["criteria"] == {"violence": "violence", "not_violence": "does not involve violence"}


def test_score_question_uses_levels():
    q = build_questions(CATS, [Mode.SCORE])["score:violence"]
    assert q == {
        "type": "score",
        "instructions": "How severe is the violence content in `text`?",
        "criteria": SCORE_LEVELS,
    }


def test_all_modes_one_question_each():
    qs = build_questions(CATS, [Mode.NOUL, Mode.CHOICE, Mode.SCORE])
    assert len(qs) == 6


def test_mode_score_extracts_each_answer_type():
    assert mode_score(Mode.NOUL, "violence", {"noul": 0.8}) == 0.8
    assert mode_score(Mode.CHOICE, "violence", {"probabilities": {"violence": 0.7, "not_violence": 0.3}}) == 0.7
    assert mode_score(Mode.SCORE, "violence", {"score": 1.5}) == pytest.approx(0.5)


def test_mode_score_clamps_to_unit_interval():
    assert mode_score(Mode.NOUL, "v", {"noul": 1.2}) == 1.0
    assert mode_score(Mode.SCORE, "v", {"score": -0.1}) == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_mode_score_rejects_non_finite(bad):
    # a NaN would compare False against any threshold and silently pass the guard
    with pytest.raises(ValueError, match="non-finite"):
        mode_score(Mode.NOUL, "violence", {"noul": bad})
