# Changelog

## 0.1.1

Long-input fixes: both guards now score the whole text or refuse it, never a silent prefix.

- **`LayaGuard` read only the first ~300 tokens.** Laya truncates its input to `max_len - head_max_len`, so
  anything after that was never scored (a threat after ~800 words of filler scored 0.009 for violence; now 0.947).
  Longer text is now split into overlapping chunks with Laya's own tokenizer, scored in one batched
  `predict_batch` call, and each category keeps its highest-scoring chunk. Short text uses the same single
  `predict` call as before.
- **`GlinerGuard` truncated any single "word" longer than a chunk**: text without spaces (CJK, base64, long URLs)
  lost everything past ~490 tokens. Such words are now split into overlapping token pieces.
- **`max_chunks` (default 64)** on both guards: text needing more chunks raises `InputTooLong` (a `ValueError`)
  instead of scoring part of it. Catch it to fail closed.
- `reflexguard.questions.strongest()` merges per-chunk answers by each question's own mode score, shared by both
  guards.
- Eval: Laya re-scored on the 40 core and 8 ood prompts past its window (`scripts/stale_long_prompts.py`); its
  results move by at most one prompt, so the published comparison stands.

## 0.1.0

First release.

- `Guard`: multi-label guardrails on typed-decision encoders. Every category gets a score in [0, 1] per mode —
  `NOUL` (yes/no), `CHOICE` (`{category, not_category}`), `SCORE` (severity) — combined with `|` (any), `&` (all)
  or `votes=k`; `threshold` is optional (`None` ranks only) and `GuardResult.at()` re-thresholds without re-running.
- `LayaGuard`: the questions on a local Laya router.
- `GlinerGuard`: GLiNER2.5-Decide on its ONNX export ([`nishparadox/gliner2.5-decide-onnx`](https://huggingface.co/nishparadox/gliner2.5-decide-onnx)),
  torch-free (`pip install 'reflexguard[gliner]'`). fp32 by default (identical to the torch model), `int8` / `fp16`
  opt-in; long text chunked to 512 tokens with the highest-scoring chunk kept per category; questions that do not fit
  one prompt split across passes; `noul_orders` (default 3) averages NOUL over rotated label orders, which lifts NOUL's AUC on the core set from 0.55 to 0.68; `batch_size` pads passes into
  one onnxruntime call (for GPU providers; 1 on CPU).
- `reflexguard.__version__`.
- Benchmark: `scripts/eval.py` (cached per-mode scores, every combination and threshold computed offline, seaborn
  figures), a core set (`data/eval`) and a public out-of-distribution set (`data/eval/ood`).
- Marimo playground (`notebooks/playground.py`) with Laya and GLiNER, both loaded on demand.

`VonGuard` and `JevGuard` are placeholders and raise `NotImplementedError`.
