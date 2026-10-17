# Experiments

Write-ups of every experiment behind reflexguard, in the order they were run. Each states the question, setup,
results and caveats, and points at the data and figures it came from. Scores are cached in `data/eval/**/cache/`, so
`scripts/eval.py report | plot | paper` reproduces every table and figure without calling a model.

| # | experiment | headline |
|---|---|---|
| [01](01-gliner-decide-spike.md) | GLiNER2.5-Decide spike | answers all three modes with full distributions; NOUL works as one multi-label question |
| [02](02-gliner-onnx-export.md) | ONNX export | fp32 identical to torch, no torch dependency; int8 2× faster but drifts up to 0.18 |
| [03](03-akd-issue12-gliner.md) | GLiNER in akd-guardrails#12 | stronger than Laya question for question; not a standalone input guard |
| [04](04-core-eval.md) | Core benchmark | Jev and GPT-6 Luna lead in every config; GLiNER best on intent twins |
| [05](05-noul-label-orders.md) | NOUL label-order bias | first label inflated on benign text; 3 rotated orders: AUC 0.55 → 0.68 |
| [06](06-speed.md) | GLiNER speed | batching, fewer passes and CoreML don't help on CPU; int8 does |
| [07](07-ood-eval.md) | Out-of-distribution benchmark | rankings transfer for the API models and GLiNER, thresholds transfer for none |
| [08](08-long-input.md) | Long input | two truncation bugs fixed; modes combine within each chunk |
| [09](09-jev-eval.md) | Jev 1.14 | best ranking on both sets; core thresholds over-block on public data |
| [10](10-openai-decisions-eval.md) | GPT-6 Luna (OpenAI Decisions) | second-best ranking, least over-blocking; refuses some `choice` / `score` questions |
| [11](11-ensembles-and-cascades.md) | Ensembles and cascades (offline) | `Votes(2)` halves OOD over-blocking; Laya → Jev at `Margin(0.2)` keeps Jev's recall with half the API calls |

Figures for papers: `data/eval/figures/paper_*.{pdf,png}` — one colour, hatch and marker per model throughout.
