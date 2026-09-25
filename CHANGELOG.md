# Changelog

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
