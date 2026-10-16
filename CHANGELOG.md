# Changelog

## 0.1.2

`JevGuard`, and how a long text's chunks become one result.

- **`JevGuard`** (`pip install 'reflexguard[jev]'`): TypeSafe Jev through OpenRouter's Decisions API. Every question
  of every mode goes in one request with the text sent once; `model` defaults to the dated snapshot
  `typesafe/jev-1.14-20261014` and a call raises if another snapshot answers; text past ~30 KB is chunked below Jev's
  long-context mode and sent concurrently; 429 / 5xx are retried; `usage` tracks requests, input tokens and cost.

- **Modes are combined within a chunk, then the worst chunk wins.** 0.1.1 took each mode's highest chunk separately
  and combined those, so `all` / `votes=k` could count two modes as agreeing when they peaked on different passages.
  Now a category's modes must agree about the same chunk. Categories still peak independently, so a text can be
  flagged for one harm in one chunk and another harm elsewhere (multi-label). `any` gives the same result either way.
- `CategoryResult.chunk` says which chunk set a category's score (its `by_mode` and `raw` come from that chunk);
  `GuardResult.chunks` says how many chunks were scored. A guard's `predict()` may return one answer dict per chunk;
  the base `Guard` does the merge, so every guard gets it.
- `GlinerGuard` chunks the text once, sized for its longest prompt, so every mode and question group scores the same
  passages.
- Chunk overlap (`overlap=`, default 32) is counted in tokens for both guards; GLiNER rounds up to whole words.
- `questions.strongest()` (0.1.1) is removed.
- Eval: prompts long enough to be chunked are re-scored, and multi-mode configs combine per chunk (`scripts/eval.py`
  caches per-chunk scores for them). Jev is evaluated under the same protocol (API calls run concurrently; each cache
  entry records the served snapshot). `eval.py paper` draws publication figures (PDF + PNG): grouped bars with a
  colour-blind-safe Okabe–Ito palette, a hatch per model for greyscale print, embedded fonts.

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
