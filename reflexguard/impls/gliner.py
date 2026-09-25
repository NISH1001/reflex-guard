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
from ._gliner_onnx import MARKERS, GlinerOnnx, Task, words

DEFAULT_MODEL = "nishparadox/gliner2.5-decide-onnx"
PRECISIONS = {"fp32": "model.onnx", "fp16": "model_fp16.onnx", "int8": "model_int8.onnx"}


class GlinerGuard(Guard):
    """Each mode is its own forward pass: sharing one prompt across modes blurs their scores.

    Text longer than `max_tokens` (prompt included) is split into overlapping chunks and each
    category keeps its highest-scoring chunk, so a violation anywhere in the text is seen.
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
        overlap: int = 32,
        threads: int | None = None,
        runtime: GlinerOnnx | None = None,
    ) -> None:
        super().__init__(categories, mode, threshold, votes, debug)
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
        if max_tokens < 1 or overlap < 0:
            raise ValueError(f"need max_tokens >= 1 and overlap >= 0, got {max_tokens} and {overlap}")
        for name, desc in self.categories.items():
            for marker in MARKERS:
                if marker in name or marker in desc:
                    raise ValueError(f"category {name!r} contains {marker!r}, a GLiNER marker token")
        self.model, self.precision, self.threads = model, precision, threads
        self.max_tokens, self.overlap = max_tokens, overlap
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
            rt = self._runtime
            text_words = words(context)
            answers: dict[str, Any] = {}
            for mode, tasks in self._tasks(questions).items():
                prompt = rt.prompt(tasks)
                chunks = self._chunks(rt, text_words, len(prompt[0]), mode)
                self._log("{}: {} prompt tokens, {} chunk(s)", mode, len(prompt[0]), len(chunks))
                for chunk in chunks:
                    probs = rt.probabilities(prompt, chunk, tasks)
                    for qid, answer in _answers(mode, probs).items():
                        if qid not in answers or _key(mode, qid, answer) > _key(mode, qid, answers[qid]):
                            answers[qid] = answer
            return answers

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

    def _chunks(self, rt: GlinerOnnx, text_words: list[str], prompt_len: int, mode: str) -> list[list[int]]:
        budget = self.max_tokens - prompt_len
        if budget < 1:
            raise ValueError(
                f"the {mode} prompt alone is {prompt_len} tokens, over max_tokens={self.max_tokens}; "
                "use fewer categories, shorter descriptions or a larger max_tokens"
            )
        word_ids = [rt.ids(w) for w in text_words]
        chunks, start = [], 0
        while True:
            end, size = start, 0
            while end < len(word_ids) and (end == start or size + len(word_ids[end]) <= budget):
                size += len(word_ids[end])
                end += 1
            chunks.append([i for w in word_ids[start:end] for i in w][:budget])
            if end >= len(word_ids):
                return chunks
            start = max(end - self.overlap, start + 1)


def _text(name: str) -> str:
    return name.replace("_", " ")


def _describe(name: str, description: str) -> str:
    # "(" and ")" are gliner2 prompt structure; keep the meaning with commas instead.
    desc = re.sub(r"\s*\(\s*", ", ", description).replace(")", "").strip(" ,")
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
