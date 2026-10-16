# 01 · GLiNER2.5-Decide spike: can it answer reflexguard's questions?

**Date:** 2026-09-25 · **Outcome:** yes, with question shapes chosen per mode; led to `GlinerGuard`.

## Question

Can [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (DeBERTa-v3-large, 340M, Apache-2.0)
answer reflexguard's three modes — `noul` (yes/no), `choice` (`{category, not_category}`), `score` (severity) — with
a probability per answer rather than only a decision?

## Setup

`gliner2==2.0.0` (`gliner2[local]`: plain `gliner2` installs no torch/transformers), `gliner2.classification`
(`Classifier`, `ClassificationSchema` with `.single`, `.multi`, `.ordinal`). 3 categories (violence, pii_exposure,
self_harm), 4 prompts (a threat, an email and phone number, a banana-bread question, a suicidal message). CPU.

## Results

`ClassificationResult.probabilities(task)` returns the **full distribution** over every label, so all three modes
get real scores.

| question shape | hits (violence / pii / self-harm) | clean text | verdict |
|---|---|---|---|
| NOUL as `yes`/`no` labels per category | knife threat → violence **0.21** | ~0.5 | unusable |
| NOUL as **one multi-label task over all categories** | **0.93 / 0.83 / 0.76** | 0.15–0.45 | best, fastest |
| NOUL as one sigmoid label per category | 0.78 / 0.70 / 0.59 | ~0.49 | weak |
| CHOICE `{violence, not_violence}` | 0.82 / 0.68 / 0.65 | ~0.3 | good |
| SCORE ordinal none…severe | arg-max level right | — | flat distributions: expected level / 3 only spans ~0.38–0.54 |

Latency, 3 categories, CPU: NOUL ~175 ms; all three modes in one schema ~480 ms vs ~1,460 ms as 9 separate calls.

## Findings that shaped `GlinerGuard`

- **Sharing one prompt across tasks shifts scores**: the knife threat's NOUL violence was 0.93 alone and 0.63 with all
  7 tasks in one schema. So `GlinerGuard` runs one pass per mode.
- `(` and `)` are reserved in gliner2 labels and instructions; reflexguard's descriptions are turned into commas.
- SCORE's flat distributions mean its thresholds don't carry over from other models.
