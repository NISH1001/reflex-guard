# 07 · Out-of-distribution benchmark: do rankings and thresholds transfer?

**Dates:** 2026-09-25 → 2026-10-16 · **Version:** reflexguard v0.2.0 · **Data:** `data/eval/ood/prompts.csv` (988
prompts) · **Full tables:** `data/eval/ood/results.md`

## Question

Do the models, and the thresholds tuned on the core set, hold up on public data none of them was tuned on?

## Set and protocol

- **988 prompts** from three CC-BY-4.0 sets: Aegis 2.0 test (200 unsafe, 99 safe), XSTest v2 (199 unsafe contrasts,
  250 safe look-alikes), OR-Bench (120 toxic, 120 hard-1k over-refusal prompts). 519 harmful, 99 benign, 370 hard
  negatives; 8 content categories mapped from each source's labels (`scripts/build_ood_data.py`).
- **Nothing is tuned here:** each config keeps its core-dev threshold. AUC is threshold-free.
- Neither model card names these sets as training data — "not named", not "provably unseen"; all predate the models.

## Results

| model | config | recall | benign FP | hard-neg FP | OOD AUC | core AUC |
|---|---|---|---|---|---|---|
| Laya | noul | 58.8% | 34.3% | 32.2% | 0.695 | 0.857 |
| Laya | any | 34.3% | 18.2% | **12.4%** | 0.662 | 0.840 |
| GLiNER fp32 | choice | 74.4% | 31.3% | 40.5% | 0.760 | 0.776 |
| GLiNER int8 | choice | 72.3% | 26.3% | 39.2% | 0.750 | 0.766 |
| Jev 1.14 | **score** | 90.2% | 24.2% | 49.5% | **0.881** | 0.940 |
| Jev 1.14 | choice | **97.1%** | 36.4% | 68.6% | 0.848 | 0.910 |
| GPT-6 Luna | score | 77.8% | 15.2% | 25.7% | 0.855 | 0.920 |
| GPT-6 Luna | noul | 73.8% | **9.1%** | 30.5% | 0.810 | 0.881 |

## Findings

- **Rankings transfer unevenly.** Laya loses 0.14–0.18 AUC in every mode; Jev and GPT-6 Luna 0.06–0.08; GLiNER's
  `choice` only ~0.015. Jev has the best OOD ranking (0.83–0.88), GPT-6 Luna next (0.81–0.86).
- **Thresholds do not transfer.** With core-dev thresholds, Jev catches 90–97% of harmful prompts but flags half or
  more of the benign look-alikes; GPT-6 Luna flags the fewest benign prompts (9–19%) at 74–85% recall; Laya misses
  the most (34–69% recall).
- XSTest's safe look-alikes are the hardest part of the set for every model.
- GPT-6 Luna's `choice` / `score` results here are partly its refusal policy (see 10).

The case this makes: calibrate thresholds per deployment on its own traffic, rather than carrying one threshold
across domains.
