"""Torch-free GLiNER2 classification on onnxruntime + tokenizers + numpy.

Rebuilds exactly what gliner2's processor feeds the encoder (inference mode, no sampling):
    ( [P] "{task}: {instruction} [DESCRIPTION] {label}: {desc} ..." ( [L] l1 [L] l2 ... ) )
    [SEP_STRUCT] <next task> ... [SEP_TEXT] <lowercased words>
Each piece is tokenized on its own, with no [CLS]/[SEP]. Label logits are read at the [L] markers.
The export and its parity check against gliner2 live at nishparadox/gliner2.5-decide-onnx.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

MARKERS = ("[P]", "[L]", "[C]", "[E]", "[R]", "[DESCRIPTION]", "[EXAMPLE]", "[OUTPUT]", "[SEP_STRUCT]", "[SEP_TEXT]")

_WORDS = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)


@dataclass(frozen=True)
class Task:
    name: str
    labels: dict[str, str]  # label -> description ("" for none)
    instruction: str = ""
    exclusive: bool = True  # softmax over labels; False = one sigmoid per label

    def pieces(self) -> list[str]:
        prompt = f"{self.name}: {self.instruction}" if self.instruction else self.name
        for label, desc in self.labels.items():
            if desc:
                prompt += f" [DESCRIPTION] {label}: {desc}"
        out = ["(", "[P]", prompt, "("]
        for label in self.labels:
            out += ["[L]", label]
        return out + [")", ")"]


def words(text: str) -> list[str]:
    """gliner2's whitespace word splitter, after its trailing-punctuation rule."""
    if not text.endswith((".", "!", "?")):
        text = text + "." if text else "."
    return [m.group().lower() for m in _WORDS.finditer(text)]


class GlinerOnnx:
    def __init__(self, session: Any, tokenizer: Any) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def load(cls, model_path: str, tokenizer_path: str, threads: int | None = None) -> GlinerOnnx:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        return cls(session, Tokenizer.from_file(tokenizer_path))

    def ids(self, piece: str) -> list[int]:
        ids = self._cache.get(piece)
        if ids is None:
            ids = self._cache[piece] = list(self.tokenizer.encode(piece, add_special_tokens=False).ids)
        return ids

    def prompt(self, tasks: list[Task]) -> tuple[list[int], list[int]]:
        """Token ids up to and including [SEP_TEXT], and the position of every [L] marker."""
        ids: list[int] = []
        positions: list[int] = []
        for t_idx, task in enumerate(tasks):
            if t_idx:
                ids += self.ids("[SEP_STRUCT]")
            for i, piece in enumerate(task.pieces()):
                if i >= 4 and i % 2 == 0 and piece == "[L]":
                    positions.append(len(ids))
                ids += self.ids(piece)
        return ids + self.ids("[SEP_TEXT]"), positions

    def scores(self, text: str, tasks: list[Task], max_tokens: int = 512, overlap: int = 32,
               label: str = "task") -> list[dict[str, dict[str, float]]]:
        """Probabilities for every task and label, one dict per chunk of `text`.

        The prompt and each chunk fit in `max_tokens` together; chunks share `overlap` words.
        """
        prompt = self.prompt(tasks)
        return [self.probabilities(prompt, chunk, tasks)
                for chunk in self.chunks(words(text), len(prompt[0]), max_tokens, overlap, label)]

    def chunks(self, text_words: list[str], prompt_len: int, max_tokens: int, overlap: int,
               label: str = "task") -> list[list[int]]:
        budget = max_tokens - prompt_len
        if budget < 1:
            raise ValueError(
                f"the {label} prompt alone is {prompt_len} tokens, over max_tokens={max_tokens}; "
                "use fewer categories, shorter descriptions or a larger max_tokens"
            )
        word_ids = [self.ids(w) for w in text_words]
        chunks, start = [], 0
        while True:
            end, size = start, 0
            while end < len(word_ids) and (end == start or size + len(word_ids[end]) <= budget):
                size += len(word_ids[end])
                end += 1
            chunks.append([i for w in word_ids[start:end] for i in w][:budget])
            if end >= len(word_ids):
                return chunks
            start = max(end - overlap, start + 1)

    def probabilities(self, prompt: tuple[list[int], list[int]], text_ids: list[int],
                      tasks: list[Task]) -> dict[str, dict[str, float]]:
        import numpy as np

        prompt_ids, positions = prompt
        ids = prompt_ids + text_ids
        (logits,) = self.session.run(["logits"], {
            "input_ids": np.asarray([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "label_positions": np.asarray([positions], dtype=np.int64),
        })
        flat = np.asarray(logits[0], dtype=np.float64)
        out, start = {}, 0
        for task in tasks:
            x = flat[start:start + len(task.labels)]
            start += len(task.labels)
            p = np.exp(x - x.max()) / np.exp(x - x.max()).sum() if task.exclusive else 1 / (1 + np.exp(-x))
            out[task.name] = dict(zip(task.labels, p.tolist()))
        return out
