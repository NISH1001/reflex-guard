"""GlinerGuard: runs guard questions on GLiNER2.5-Decide through onnxruntime (no torch)."""

from __future__ import annotations

import asyncio
import os
import re
import threading
from typing import Any

from ..guard import Categories, Guard
from ..modes import Expr, Mode
from ..questions import SCORE_LEVELS
from ..result import Threshold
from ._gliner_onnx import MARKERS, GlinerOnnx, Task

DEFAULT_MODEL = "nishparadox/gliner2.5-decide-onnx"
PRECISIONS = {"fp32": "model.onnx", "fp16": "model_fp16.onnx", "int8": "model_int8.onnx"}


class GlinerGuard(Guard):
    """Each mode is its own forward pass: sharing one prompt across modes blurs their scores.

    A mode whose questions would take more than `prompt_tokens` (default: half of `max_tokens`)
    is split across several passes, leaving room for the text. Text longer than what is left of
    `max_tokens` is split into overlapping chunks and each category keeps its highest-scoring
    chunk, so a violation anywhere in the text is seen.

    NOUL's multi-label question is sensitive to the order of its labels; `noul_orders=k` asks it
    k times with the categories rotated and averages each category's score. `batch_size` pads
    passes into one onnxruntime call, which pays off on GPU providers; on CPU a single pass
    already uses every core, so the default is 1.
    """

    def __init__(
        self,
        categories: Categories,
        mode: Expr = Mode.NOUL,
        threshold: Threshold = None,
        votes: int | None = None,
        debug: bool = False,
        model: str = DEFAULT_MODEL,
        precision: str = "fp32",
        max_tokens: int = 512,
        prompt_tokens: int | None = None,
        overlap: int = 32,
        noul_orders: int = 3,
        batch_size: int = 1,
        threads: int | None = None,
        runtime: GlinerOnnx | None = None,
    ) -> None:
        super().__init__(categories, mode, threshold, votes, debug)
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
        prompt_tokens = max_tokens // 2 if prompt_tokens is None else prompt_tokens
        if max_tokens < 1 or overlap < 0 or batch_size < 1 or noul_orders < 1 or not 0 < prompt_tokens < max_tokens:
            raise ValueError(
                "need max_tokens >= 1, 0 < prompt_tokens < max_tokens, overlap >= 0, batch_size >= 1 and "
                f"noul_orders >= 1, got {max_tokens}, {prompt_tokens}, {overlap}, {batch_size} and {noul_orders}"
            )
        for name, desc in self.categories.items():
            for marker in MARKERS:
                if marker in name or marker in desc:
                    raise ValueError(f"category {name!r} contains {marker!r}, a GLiNER marker token")
        self.model, self.precision, self.threads = model, precision, threads
        self.max_tokens, self.prompt_tokens, self.overlap = max_tokens, prompt_tokens, overlap
        self.batch_size, self.noul_orders = batch_size, noul_orders
        self._runtime = runtime
        self._lock = threading.Lock()

    async def predict(self, context: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return await asyncio.to_thread(self._predict_sync, context, questions)

    def _predict_sync(self, context: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        # One lock: the runtime is created once and calls into the model are serialized.
        with self._lock:
            if self._runtime is None:
                self._log("loading {} ({})", self.model, self.precision)
                self._runtime = load_runtime(self.model, self.precision, self.threads)
            # Every pass of every mode goes through one batched run: modes never share a prompt,
            # they only share the batch. A slot is one (question group, chunk) of a mode; NOUL's
            # label orders fill the same slots, since rotating labels keeps the prompt length.
            slots, requests = [], []
            for mode, tasks in self._tasks(questions).items():
                orders = self.noul_orders if mode == Mode.NOUL.name else 1
                for k in range(orders):
                    rotated = [_rotate(t, k, orders) for t in tasks]
                    planned = [r for group in self._groups(rotated)
                               for r in self._runtime.plan(context, group, self.max_tokens, self.overlap, mode)]
                    slots += [(mode, i) for i in range(len(planned))]
                    requests += planned
                self._log("{}: {} pass(es)", mode, sum(m == mode for m, _ in slots))
            merged: dict[tuple[str, int], list[dict[str, dict[str, float]]]] = {}
            for slot, probs in zip(slots, self._runtime.run(requests, self.batch_size)):
                merged.setdefault(slot, []).append(probs)
            answers: dict[str, Any] = {}
            for (mode, _), per_order in merged.items():
                for qid, answer in _answers(mode, _mean(per_order)).items():
                    if qid not in answers or _key(mode, qid, answer) > _key(mode, qid, answers[qid]):
                        answers[qid] = answer
            return answers

    def _groups(self, tasks: list[Task]) -> list[list[Task]]:
        """Pack tasks in order into passes whose prompt stays within `prompt_tokens`."""
        groups: list[list[Task]] = []
        for task in tasks:
            if groups and len(self._runtime.prompt([*groups[-1], task])[0]) <= self.prompt_tokens:
                groups[-1].append(task)
            else:
                groups.append([task])
        return groups

    def _tasks(self, questions: dict[str, dict[str, Any]]) -> dict[str, list[Task]]:
        by_mode: dict[str, list[str]] = {}
        for qid in questions:
            mode, _, name = qid.partition(":")
            by_mode.setdefault(mode, []).append(name)
        tasks: dict[str, list[Task]] = {}
        for mode, names in by_mode.items():
            if mode == Mode.NOUL.name:
                labels = {n: _describe(n, self.categories[n]) for n in names}
                tasks[mode] = [Task("noul", labels, "Which of these does the text involve?", exclusive=False)]
            elif mode == Mode.CHOICE.name:
                tasks[mode] = [
                    Task(f"choice:{n}", {n: _describe(n, self.categories[n]), f"not_{n}": f"does not involve {_text(n)}"},
                         f"Does the text involve {_text(n)}?")
                    for n in names
                ]
            elif mode == Mode.SCORE.name:
                tasks[mode] = [
                    Task(f"score:{n}", dict.fromkeys(SCORE_LEVELS, ""),
                         f"How severe is the {_describe(n, self.categories[n])} content in the text?")
                    for n in names
                ]
            else:
                raise ValueError(f"GlinerGuard does not support mode {mode!r}")
        return tasks


def _rotate(task: Task, k: int, orders: int) -> Task:
    """The task with its labels rotated by k/orders of their length (k = 0 leaves it as is)."""
    labels = list(task.labels.items())
    shift = k * len(labels) // orders
    return Task(task.name, dict(labels[shift:] + labels[:shift]), task.instruction, task.exclusive)


def _mean(per_order: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Average each task's label probabilities over the label orders it was asked in."""
    if len(per_order) == 1:
        return per_order[0]
    return {task: {label: sum(p[task][label] for p in per_order) / len(per_order) for label in labels}
            for task, labels in per_order[0].items()}


def _text(name: str) -> str:
    return name.replace("_", " ")


def clean(text: str) -> str:
    """Drop "(" and ")", which are gliner2 prompt structure, keeping the meaning with commas."""
    return re.sub(r"\s*\(\s*", ", ", text).replace(")", "").strip(" ,")


def _describe(name: str, description: str) -> str:
    desc = clean(description)
    return f"{_text(name)}: {desc}" if desc else _text(name)


def _answers(mode: str, probs: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    """One chunk's probabilities -> answers keyed by question id, in the shape `mode_score` reads."""
    if mode == Mode.NOUL.name:
        return {f"noul:{n}": {"noul": p} for n, p in probs["noul"].items()}
    if mode == Mode.CHOICE.name:
        return {task: {"probabilities": dict(p)} for task, p in probs.items()}
    return {
        # Same shape as Laya's score answers: probabilities keyed by level index.
        task: {"score": sum(i * p[level] for i, level in enumerate(SCORE_LEVELS)),
               "probabilities": {str(i): p[level] for i, level in enumerate(SCORE_LEVELS)}}
        for task, p in probs.items()
    }


def _key(mode: str, qid: str, answer: dict[str, Any]) -> float:
    """How strongly one chunk's answer points at the category; the strongest chunk wins."""
    if mode == Mode.NOUL.name:
        return answer["noul"]
    if mode == Mode.CHOICE.name:
        return answer["probabilities"][qid.partition(":")[2]]
    return answer["score"]


def load_runtime(model: str = DEFAULT_MODEL, precision: str = "fp32", threads: int | None = None) -> GlinerOnnx:
    """Load the ONNX model once, to share across guards with `GlinerGuard(runtime=...)`."""
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError as e:
        raise ImportError("GlinerGuard needs onnxruntime and tokenizers: pip install 'reflexguard[gliner]'") from e
    if os.path.isdir(model):
        model_path = os.path.join(model, PRECISIONS[precision])
        tokenizer_path = os.path.join(model, "tokenizer.json")
    else:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(model, PRECISIONS[precision])
        tokenizer_path = hf_hub_download(model, "tokenizer.json")
    return GlinerOnnx.load(model_path, tokenizer_path, threads)
