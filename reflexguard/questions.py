"""Turn categories and modes into System One questions, and answers back into scores."""

from __future__ import annotations

from typing import Any, Iterable

from .modes import Mode

SCORE_LEVELS = ["none", "minor", "serious", "severe"]


def label(name: str, description: str = "") -> str:
    """`pii_exposure`, `names, emails` -> `pii exposure (names, emails)`."""
    text = name.replace("_", " ")
    return f"{text} ({description})" if description else text


def question_id(mode: Mode, category: str) -> str:
    return f"{mode.name}:{category}"


def build_questions(categories: dict[str, str], modes: Iterable[Mode]) -> dict[str, dict[str, Any]]:
    """One question per (mode, category), keyed `"{mode}:{category}"`."""
    questions = {}
    for mode in modes:
        for name, description in categories.items():
            lbl = label(name, description)
            if mode == Mode.NOUL:
                q = {"type": "noul", "instructions": f"Does `text` involve {lbl}?"}
            elif mode == Mode.CHOICE:
                q = {
                    "type": "choice",
                    "instructions": f"Does `text` involve {lbl}?",
                    "criteria": {name: lbl, f"not_{name}": f"does not involve {lbl}"},
                }
            elif mode == Mode.SCORE:
                q = {
                    "type": "score",
                    "instructions": f"How severe is the {lbl} content in `text`?",
                    "criteria": list(SCORE_LEVELS),
                }
            else:
                raise ValueError(f"unknown mode {mode!r}")
            questions[question_id(mode, name)] = q
    return questions


def mode_score(mode: Mode, category: str, answer: dict[str, Any]) -> float:
    """A model answer -> this category's score in [0, 1] for that mode."""
    if mode == Mode.NOUL:
        value = answer["noul"]
    elif mode == Mode.CHOICE:
        value = answer["probabilities"][category]
    elif mode == Mode.SCORE:
        value = answer["score"] / (len(SCORE_LEVELS) - 1)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return min(max(float(value), 0.0), 1.0)
